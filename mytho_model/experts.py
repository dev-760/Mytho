"""
Mixture-of-Experts (MoE) layer with top-k gating and load-balancing loss.

Each expert is a SwiGLU feed-forward network.  A lightweight router selects
the top-k experts per token and the outputs are combined with normalised
gating weights.

Auxiliary load-balancing loss (Shazeer et al., 2017; Fedus et al., 2022)
encourages uniform expert utilisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythoConfig
from .components import SwiGLU


class TopKRouter(nn.Module):
    """Differentiable top-k routing with jitter noise during training."""

    def __init__(self, d_model: int, n_experts: int, n_active: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
        Returns:
            topk_weights: [B, S, k]  – normalised weights
            topk_indices: [B, S, k]  – expert indices
            router_logits: [B, S, E] – raw logits (for aux loss)
        """
        logits = self.gate(x)                                 # [B, S, E]

        # Add small noise during training for exploration
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)                     # [B, S, E]
        topk_w, topk_i = torch.topk(probs, self.n_active, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_w, topk_i, logits


def load_balancing_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """
    Compute the auxiliary load-balancing loss.

    L_bal = n_experts * Σ_i (f_i · p_i)
    where f_i = fraction of tokens routed to expert i
          p_i = mean router probability for expert i
    """
    probs = F.softmax(router_logits, dim=-1)          # [B, S, E]
    B, S, E = probs.shape

    # f_i: fraction of tokens dispatched to each expert
    one_hot = F.one_hot(topk_indices, n_experts).float()   # [B,S,k,E]
    f = one_hot.sum(dim=2).mean(dim=(0, 1))                # [E]

    # p_i: mean probability assigned to each expert
    p = probs.mean(dim=(0, 1))                              # [E]

    return E * (f * p).sum()


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer.

    Each token is routed to ``n_active_experts`` out of ``n_experts``
    SwiGLU feed-forward experts.
    """

    def __init__(self, config: MythoConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_active = config.n_active_experts
        self.balance_coeff = config.expert_balance_coeff

        self.router = TopKRouter(config.d_model, config.n_experts, config.n_active_experts)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.d_model, config.d_expert_ff, config.dropout)
                for _ in range(config.n_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, S, D]
        Returns:
            output:   [B, S, D]
            aux_loss: scalar load-balancing loss
        """
        B, S, D = x.shape
        topk_w, topk_i, logits = self.router(x)   # weights/indices [B,S,k]

        # ── Dispatch & combine ──────────────────────────────────────
        output = torch.zeros_like(x)

        # Flatten for efficient expert computation
        flat_x = x.view(-1, D)                          # [B*S, D]
        flat_w = topk_w.view(-1, self.n_active)          # [B*S, k]
        flat_i = topk_i.view(-1, self.n_active)          # [B*S, k]
        flat_out = torch.zeros_like(flat_x)              # [B*S, D]

        for k_idx in range(self.n_active):
            expert_indices = flat_i[:, k_idx]             # [B*S]
            expert_weights = flat_w[:, k_idx : k_idx + 1]  # [B*S, 1]

            for e_id in range(self.n_experts):
                mask = expert_indices == e_id              # [B*S]
                if not mask.any():
                    continue
                expert_input = flat_x[mask]               # [n_tok, D]
                expert_out = self.experts[e_id](expert_input)
                flat_out[mask] += expert_weights[mask] * expert_out

        output = flat_out.view(B, S, D)

        # ── Auxiliary loss ──────────────────────────────────────────
        aux_loss = load_balancing_loss(logits, topk_i, self.n_experts)
        return output, aux_loss * self.balance_coeff


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Switch Transformer Router — Fedus et al. (2022)
#  Reference: https://arxiv.org/abs/2101.03961
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SwitchRouter(nn.Module):
    """
    Top-1 routing with *capacity factor* and *router z-loss*.

    Key improvements over vanilla top-k:
      • Each token is sent to exactly **one** expert (top-1) for efficiency.
      • A *capacity factor* limits how many tokens each expert processes,
        dropping overflow tokens to maintain load balance.
      • A *z-loss* on router logits prevents them from growing unboundedly.
    """

    def __init__(self, d_model: int, n_experts: int,
                 capacity_factor: float = 1.25, z_loss_coeff: float = 1e-3):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.z_loss_coeff = z_loss_coeff
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dispatch_mask: [B*S, E] bool — which expert each token goes to
            combine_weights: [B*S, E] float — gating weight per expert
            router_logits: [B, S, E]
            z_loss: scalar
        """
        B, S, D = x.shape
        N = B * S
        logits = self.gate(x)                                     # [B, S, E]

        # Router z-loss: penalise large logits for stability
        z_loss = self.z_loss_coeff * (logits.float() ** 2).mean()

        probs = F.softmax(logits, dim=-1)                         # [B, S, E]
        flat_probs = probs.view(N, -1)                            # [N, E]

        # Top-1 selection
        expert_idx = flat_probs.argmax(dim=-1)                    # [N]
        expert_weight = flat_probs.gather(1, expert_idx.unsqueeze(1))  # [N, 1]

        # Capacity: max tokens per expert
        capacity = int(self.capacity_factor * N / self.n_experts)

        dispatch_mask = torch.zeros(N, self.n_experts, device=x.device, dtype=torch.bool)
        combine_weights = torch.zeros(N, self.n_experts, device=x.device)

        for e in range(self.n_experts):
            assigned = (expert_idx == e).nonzero(as_tuple=True)[0]
            if len(assigned) > capacity:
                # Keep the top-scoring tokens, drop the rest
                scores = flat_probs[assigned, e]
                _, keep_idx = scores.topk(capacity)
                assigned = assigned[keep_idx]
            dispatch_mask[assigned, e] = True
            combine_weights[assigned, e] = expert_weight[assigned, 0]

        return dispatch_mask, combine_weights, logits, z_loss


class SwitchMoELayer(nn.Module):
    """
    Switch Transformer MoE: top-1 routing with capacity factor.

    More parameter-efficient than top-k MoE — each token is processed
    by exactly one expert, enabling scaling to trillions of parameters.
    """

    def __init__(self, config: MythoConfig, capacity_factor: float = 1.25):
        super().__init__()
        self.n_experts = config.n_experts
        self.balance_coeff = config.expert_balance_coeff

        self.router = SwitchRouter(
            config.d_model, config.n_experts, capacity_factor
        )
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.d_expert_ff, config.dropout)
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        N = B * S
        flat_x = x.view(N, D)

        dispatch, weights, logits, z_loss = self.router(x)
        flat_out = torch.zeros_like(flat_x)

        for e in range(self.n_experts):
            mask = dispatch[:, e]
            if not mask.any():
                continue
            e_in = flat_x[mask]
            e_out = self.experts[e](e_in)
            flat_out[mask] += weights[mask, e:e+1] * e_out

        output = flat_out.view(B, S, D)
        bal_loss = load_balancing_loss(logits, logits.argmax(-1).unsqueeze(-1),
                                       self.n_experts)
        return output, bal_loss * self.balance_coeff + z_loss

