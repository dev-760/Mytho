"""
Full Recurrent-Depth Transformer (Mytho) model.

Architecture overview
─────────────────────
  Tokens ──► Embedding ──► x₀
                              │
      ┌───────────────────────┘
      │   for depth t = 1 … T (adaptive):
      │     x_t = x_{t-1} + depth_emb[t]
      │     x_t = TransformerBlock(x_t)        ← weight-shared
      │     h_t = σ(halt_linear(x_t))          ← ACT halting
      │     accumulate weighted output
      │     if all tokens halted: break
      └───────────────────────┐
                              │
  Output ◄── LM Head ◄── RMSNorm ◄── ACT-weighted sum
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythoConfig
from .components import RMSNorm, precompute_rope_frequencies
from .attention import MultiLatentAttention
from .experts import MoELayer
from .memory import MemoryManager
from .uncertainty import EnsembleHead
from .scratchpad import LatentScratchpad
from .verifier import VerifierHead
from .branching import BranchingController


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Transformer Block (single block that is applied recurrently)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MythoBlock(nn.Module):
    """
    One transformer block: MLA attention → MoE FFN.

    Supports optional scratchpad:
      • Attention reads scratchpad context before self-attention
      • Experts write to scratchpad after MoE FFN
    """

    def __init__(self, config: MythoConfig, scratchpad: LatentScratchpad | None = None):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MultiLatentAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = MoELayer(config)
        self.scratchpad = scratchpad

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        scratch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            x:        updated hidden states  [B, S, D]
            aux_loss: MoE auxiliary loss      scalar
            scratch:  updated scratchpad or None
        """
        # Read scratchpad into attention context
        if self.scratchpad is not None and scratch is not None:
            scratch_ctx = self.scratchpad.read(x, scratch)
            x = x + scratch_ctx

        # Pre-norm attention with residual
        h = self.attn_norm(x)
        h = self.attn(h, rope_cos, rope_sin, mask)
        x = x + h

        # Pre-norm MoE FFN with residual
        h = self.ffn_norm(x)
        h, aux_loss = self.moe(h)
        x = x + h

        # Experts write to scratchpad
        if self.scratchpad is not None and scratch is not None:
            scratch = self.scratchpad.write(x, scratch)

        return x, aux_loss, scratch


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Adaptive Computation Time (ACT) controller
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AdaptiveComputationController(nn.Module):
    """
    Implements Adaptive Computation Time (Graves, 2016).

    At each recurrent depth step:
      1. Compute per-token halting probability  h_t = σ(W · x_t + b)
      2. Accumulate cumulative halting probability
      3. When cumulative ≥ threshold, mark token as halted
      4. Final output = Σ_t (effective_weight_t × state_t)
      5. Ponder cost = mean(n_steps_per_token) — penalises excess depth
    """

    def __init__(self, config: MythoConfig):
        super().__init__()
        self.max_depth = config.max_depth
        self.threshold = config.act_threshold
        self.act_loss_coeff = config.act_loss_coeff

        # Halting probability predictor
        self.halt_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.SiLU(),
            nn.Linear(config.d_model // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        block: MythoBlock,
        depth_embeddings: nn.Embedding,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:                input hidden states [B, S, D]
            block:            the shared MythoBlock
            depth_embeddings: nn.Embedding(max_depth, D)
            rope_cos/sin:     precomputed RoPE tables
            mask:             causal attention mask

        Returns:
            output:     ACT-weighted output [B, S, D]
            act_loss:   ponder cost (scalar)
            moe_loss:   accumulated MoE auxiliary loss (scalar)
            n_updates:  mean number of computation steps (scalar)
        """
        B, S, D = x.shape
        device = x.device

        # Tracking tensors
        halted = torch.zeros(B, S, 1, device=device, dtype=torch.bool)
        cumulative_halt = torch.zeros(B, S, 1, device=device)
        accumulated_output = torch.zeros(B, S, D, device=device)
        n_updates = torch.zeros(B, S, 1, device=device)
        total_moe_loss = torch.tensor(0.0, device=device)

        for step in range(self.max_depth):
            d_emb = depth_embeddings(
                torch.tensor(step, device=device)
            ).unsqueeze(0).unsqueeze(0)
            x_step = x + d_emb

            x_step, moe_aux, _ = block(x_step, rope_cos, rope_sin, mask)
            total_moe_loss = total_moe_loss + moe_aux

            p = torch.sigmoid(self.halt_proj(x_step))
            still_running = (~halted).float()
            p = p * still_running
            new_halted = ((cumulative_halt + p) >= self.threshold) & (~halted)
            remainder = (1.0 - cumulative_halt) * new_halted.float()
            p_effective = torch.where(new_halted, remainder, p * still_running)

            accumulated_output = accumulated_output + p_effective * x_step
            cumulative_halt = cumulative_halt + p_effective
            n_updates = n_updates + still_running
            halted = halted | new_halted
            x = x_step

            if halted.all():
                break

        not_halted_mask = (~halted).float()
        if not_halted_mask.any():
            leftover = (1.0 - cumulative_halt) * not_halted_mask
            accumulated_output = accumulated_output + leftover * x
            n_updates = n_updates + not_halted_mask

        ponder_cost = n_updates.float().mean() * self.act_loss_coeff
        return (
            accumulated_output, ponder_cost,
            total_moe_loss / max(step + 1, 1), n_updates.float().mean(),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Uncertainty-Driven ACT with Scratchpad, Verifier, and Branching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class UncertaintyDrivenACT(nn.Module):
    """
    Advanced ACT that halts based on verifier confidence + routing entropy
    + branch disagreement, not just a learned sigmoid.

    Integrates:
      - Latent scratchpad (persistent workspace)
      - Verifier head (confidence/quality/uncertainty)
      - Branching recurrence (latent beam search)
    """

    def __init__(self, config: MythoConfig, d_scratch: int = 128,
                 n_branches: int = 2, branch_depths: list | None = None):
        super().__init__()
        self.max_depth = config.max_depth
        self.act_loss_coeff = config.act_loss_coeff
        self.d_scratch = d_scratch
        self.n_branches = n_branches
        self.branch_depths = set(branch_depths or [config.max_depth // 3])

        self.scratchpad = LatentScratchpad(config.d_model, d_scratch)
        self.verifier = VerifierHead(config.d_model, d_scratch)
        self.brancher = BranchingController(config.d_model, n_branches)

        # Fallback halting (combined with verifier signals)
        self.halt_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.SiLU(), nn.Linear(config.d_model // 4, 1),
        )

    def forward(self, x, block, depth_emb, rope_cos, rope_sin, mask=None):
        B, S, D = x.shape
        device = x.device

        halted = torch.zeros(B, S, 1, device=device, dtype=torch.bool)
        cumulative_halt = torch.zeros(B, S, 1, device=device)
        accumulated_output = torch.zeros(B, S, D, device=device)
        n_updates = torch.zeros(B, S, 1, device=device)
        total_moe_loss = torch.tensor(0.0, device=device)

        scratch = self.scratchpad.init_scratch(B, S, device)
        verifier_signals = None

        for step in range(self.max_depth):
            d_emb = depth_emb(
                torch.tensor(step, device=device)
            ).unsqueeze(0).unsqueeze(0)
            x_step = x + d_emb

            # ── Branching at designated depths ──────────────────────
            if step in self.branch_depths and self.n_branches > 1:
                branches = self.brancher.branch(x_step)
                branch_results = []
                branch_scores = []
                branch_scratches = []
                for br in branches:
                    br_out, moe_aux, br_scratch = block(
                        br, rope_cos, rope_sin, mask, scratch.clone()
                    )
                    total_moe_loss = total_moe_loss + moe_aux
                    v_sig = self.verifier(br_out, br_scratch)
                    branch_results.append(br_out)
                    branch_scores.append(v_sig["confidence"])
                    branch_scratches.append(br_scratch)
                # Select best branch
                x_step = self.brancher.select_soft(branch_results, branch_scores)
                best_idx = torch.cat(branch_scores, -1).argmax(-1, keepdim=True)
                best_idx_exp = best_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_scratch)
                stacked_s = torch.stack(branch_scratches, dim=2)
                scratch = stacked_s.gather(2, best_idx_exp).squeeze(2)
            else:
                x_step, moe_aux, scratch = block(
                    x_step, rope_cos, rope_sin, mask, scratch
                )
                total_moe_loss = total_moe_loss + moe_aux

            # ── Verifier-driven halting ─────────────────────────────
            verifier_signals = self.verifier(x_step, scratch)
            base_halt = torch.sigmoid(self.halt_proj(x_step))
            confidence = verifier_signals["confidence"]
            uncertainty = verifier_signals["uncertainty"]
            # Combined: high confidence OR low uncertainty → halt
            p = base_halt * 0.4 + confidence * 0.4 + (1 - uncertainty) * 0.2

            still_running = (~halted).float()
            p = p * still_running
            new_halted = ((cumulative_halt + p) >= 0.99) & (~halted)
            remainder = (1.0 - cumulative_halt) * new_halted.float()
            p_effective = torch.where(new_halted, remainder, p * still_running)

            accumulated_output = accumulated_output + p_effective * x_step
            cumulative_halt = cumulative_halt + p_effective
            n_updates = n_updates + still_running
            halted = halted | new_halted
            x = x_step

            if halted.all():
                break

        not_halted = (~halted).float()
        if not_halted.any():
            accumulated_output = accumulated_output + (1 - cumulative_halt) * not_halted * x
            n_updates = n_updates + not_halted

        ponder_cost = n_updates.float().mean() * self.act_loss_coeff

        extra = {}
        if verifier_signals:
            extra["confidence"] = verifier_signals["confidence"].mean()
            extra["uncertainty"] = verifier_signals["uncertainty"].mean()

        return (
            accumulated_output, ponder_cost,
            total_moe_loss / max(step + 1, 1),
            n_updates.float().mean(), scratch, extra,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Full Mytho Language Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MythoModel(nn.Module):
    """
    Recurrent-Depth Transformer language model.

    Combines:
      • Token + depth embeddings
      • Weight-shared MythoBlock (MLA + MoE)
      • Adaptive Computation Time for dynamic depth
      • Tied input/output embeddings
    """

    def __init__(self, config: MythoConfig, use_memory: bool = False,
                 n_ensemble_heads: int = 0, use_scratchpad: bool = False,
                 d_scratch: int = 128, use_branching: bool = False,
                 n_branches: int = 2):
        super().__init__()
        self.config = config
        self.use_scratchpad = use_scratchpad

        # ── Embeddings ──────────────────────────────────────────────
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.depth_emb = nn.Embedding(config.max_depth, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # ── Scratchpad (shared by block and ACT) ────────────────────
        scratchpad = LatentScratchpad(config.d_model, d_scratch) if use_scratchpad else None

        # ── Shared transformer block(s) ────────────────────────────
        self.blocks = nn.ModuleList(
            [MythoBlock(config, scratchpad=scratchpad) for _ in range(config.n_unique_blocks)]
        )

        # ── Adaptive computation controller ────────────────────────
        if use_scratchpad:
            self.act = UncertaintyDrivenACT(
                config, d_scratch=d_scratch,
                n_branches=n_branches if use_branching else 1,
            )
        else:
            self.act = AdaptiveComputationController(config)

        # ── MemGPT-style hierarchical memory (optional) ────────────
        self.use_memory = use_memory
        self.memory = MemoryManager(config) if use_memory else None

        # ── Output head ────────────────────────────────────────────
        self.out_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # ── Deep Ensemble uncertainty head (optional) ──────────────
        self.ensemble_head = (
            EnsembleHead(n_ensemble_heads, config.d_model, config.vocab_size)
            if n_ensemble_heads > 1 else None
        )

        # ── Precompute RoPE frequencies ────────────────────────────
        rope_cos, rope_sin = precompute_rope_frequencies(
            config.d_rope, config.max_seq_len, config.rope_base,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self._init_weights)

    # ── Weight initialisation ───────────────────────────────────────
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ── Causal mask ─────────────────────────────────────────────────
    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular boolean mask [1, 1, S, S]."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    # ── Forward pass ────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, S]  token indices
            labels:    [B, S]  target token indices (shifted internally)

        Returns dict with keys:
            logits, loss, act_loss, moe_loss, mean_depth
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embed tokens
        x = self.emb_drop(self.token_emb(input_ids))             # [B, S, D]

        # Causal mask
        mask = self._make_causal_mask(S, device)

        # ── Hierarchical memory augmentation (MemGPT) ───────────────
        if self.use_memory and self.memory is not None:
            x = self.memory(x)

        # ── Recurrent depth with ACT ────────────────────────────────
        block = self.blocks[0]
        act_out = self.act(
            x, block, self.depth_emb, self.rope_cos, self.rope_sin, mask
        )
        if self.use_scratchpad:
            x, act_loss, moe_loss, mean_depth, scratch, extra = act_out
        else:
            x, act_loss, moe_loss, mean_depth = act_out
            scratch, extra = None, {}

        # ── Language model head ─────────────────────────────────────
        x = self.out_norm(x)
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "act_loss": act_loss,
            "moe_loss": moe_loss,
            "mean_depth": mean_depth,
        }
        result.update(extra)

        # ── Ensemble uncertainty (if enabled) ───────────────────────
        if self.ensemble_head is not None:
            ens = self.ensemble_head(x)
            result["ensemble_logits"] = ens["mean_logits"]
            result["disagreement"] = ens["disagreement"]

        # ── Cross-entropy loss ──────────────────────────────────────
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            total_loss = ce_loss + act_loss + moe_loss
            result["ce_loss"] = ce_loss
            result["loss"] = total_loss

        return result

    # ── Parameter count utility ─────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (not trainable_only or p.requires_grad)
        )

    # ── Autoregressive generation ───────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Greedy / sampling generation loop.
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max context window
            context = generated[:, -self.config.max_seq_len :]
            out = self.forward(context)
            next_logits = out["logits"][:, -1, :]             # [B, V]

            # Temperature scaling
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

