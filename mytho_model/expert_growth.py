"""
Expert Specialization Metrics + Dynamic Expert Growth.

Monitoring:
  • expert_entropy:     how uniformly tokens are distributed across experts
  • expert_overlap:     how often the same experts are co-selected
  • routing_stability:  how much routing changes between depth steps
  • expert_similarity:  cosine similarity between expert weight matrices

Dynamic Growth:
  • split overloaded experts (those receiving too many tokens)
  • prune dead experts (those receiving near-zero traffic)
  • merge duplicate experts (those with high weight similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experts import MoELayer


class ExpertMetrics:
    """
    Tracks and computes expert specialization metrics during training.

    Usage:
        metrics = ExpertMetrics(n_experts=8)
        # In training loop:
        metrics.update(router_logits, topk_indices)
        # Periodically:
        report = metrics.compute()
    """

    def __init__(self, n_experts: int, window: int = 100):
        self.n_experts = n_experts
        self.window = window
        self._routing_history: list[torch.Tensor] = []
        self._logit_history: list[torch.Tensor] = []

    def update(self, router_logits: torch.Tensor, topk_indices: torch.Tensor):
        """Record a routing decision. Inputs are detached automatically."""
        self._routing_history.append(topk_indices.detach().cpu())
        self._logit_history.append(router_logits.detach().cpu())
        if len(self._routing_history) > self.window:
            self._routing_history.pop(0)
            self._logit_history.pop(0)

    def expert_load(self) -> torch.Tensor:
        """Fraction of tokens routed to each expert [E]."""
        if not self._routing_history:
            return torch.zeros(self.n_experts)
        all_idx = torch.cat([r.view(-1) for r in self._routing_history])
        counts = torch.bincount(all_idx, minlength=self.n_experts).float()
        return counts / counts.sum()

    def expert_entropy(self) -> float:
        """Shannon entropy of expert load distribution. Higher = more balanced."""
        load = self.expert_load()
        load = load.clamp(min=1e-12)
        return -(load * load.log()).sum().item()

    def expert_overlap(self) -> float:
        """Mean pairwise co-selection rate (for top-k > 1)."""
        if not self._routing_history:
            return 0.0
        total, overlap = 0, 0
        for routing in self._routing_history[-20:]:
            flat = routing.view(-1, routing.shape[-1])
            if flat.shape[-1] < 2:
                continue
            for i in range(flat.shape[0]):
                experts = flat[i].tolist()
                pairs = [(experts[a], experts[b])
                         for a in range(len(experts))
                         for b in range(a+1, len(experts))]
                for a, b in pairs:
                    total += 1
                    overlap += int(a == b)
        return overlap / max(total, 1)

    def routing_stability(self) -> float:
        """How consistent routing is across recent steps (0=chaotic, 1=stable)."""
        if len(self._routing_history) < 2:
            return 1.0
        loads = []
        for routing in self._routing_history[-20:]:
            flat = routing.view(-1)
            load = torch.bincount(flat, minlength=self.n_experts).float()
            load = load / load.sum()
            loads.append(load)
        stacked = torch.stack(loads)
        stability = 1.0 - stacked.std(dim=0).mean().item()
        return max(0.0, stability)

    def expert_similarity(self, moe_layer: "MoELayer") -> torch.Tensor:
        """Pairwise cosine similarity between expert FFN weights [E, E]."""
        weight_vecs = []
        for expert in moe_layer.experts:
            params = torch.cat([p.data.view(-1) for p in expert.parameters()])
            weight_vecs.append(params)
        stacked = torch.stack(weight_vecs)
        normed = F.normalize(stacked, dim=-1)
        return torch.mm(normed, normed.t())

    def compute(self, moe_layer: "MoELayer | None" = None) -> dict:
        """Compute all metrics."""
        report = {
            "expert_load": self.expert_load().tolist(),
            "expert_entropy": self.expert_entropy(),
            "max_entropy": math.log(self.n_experts),
            "expert_overlap": self.expert_overlap(),
            "routing_stability": self.routing_stability(),
        }
        if moe_layer is not None:
            sim = self.expert_similarity(moe_layer)
            # Off-diagonal mean (similarity between different experts)
            mask = ~torch.eye(self.n_experts, dtype=torch.bool)
            report["mean_expert_similarity"] = sim[mask].mean().item()
            report["max_expert_similarity"] = sim[mask].max().item()
        return report


class DynamicExpertGrowth(nn.Module):
    """
    Neural evolution for experts: split, prune, and merge during training.

    Policies:
      • SPLIT: if an expert receives > split_threshold fraction of tokens,
        duplicate it with noise and halve routing weights.
      • PRUNE: if an expert receives < prune_threshold fraction, zero it
        and redistribute its capacity.
      • MERGE: if two experts have cosine similarity > merge_threshold,
        average their weights into one and reinitialize the other.
    """

    def __init__(
        self,
        split_threshold: float = 0.3,
        prune_threshold: float = 0.01,
        merge_threshold: float = 0.95,
        noise_scale: float = 0.01,
    ):
        super().__init__()
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.noise_scale = noise_scale

    @torch.no_grad()
    def split_expert(self, moe_layer: "MoELayer", expert_idx: int,
                     target_idx: int):
        """Copy expert_idx into target_idx with noise."""
        src = moe_layer.experts[expert_idx]
        dst = moe_layer.experts[target_idx]
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(sp.data + torch.randn_like(sp.data) * self.noise_scale)
        # Halve the router weights for the split expert
        with torch.no_grad():
            moe_layer.router.gate.weight.data[expert_idx] *= 0.5
            moe_layer.router.gate.weight.data[target_idx] = (
                moe_layer.router.gate.weight.data[expert_idx].clone()
            )

    @torch.no_grad()
    def prune_expert(self, moe_layer: "MoELayer", expert_idx: int):
        """Zero out a dead expert's router weight (soft removal)."""
        moe_layer.router.gate.weight.data[expert_idx] *= 0.0

    @torch.no_grad()
    def merge_experts(self, moe_layer: "MoELayer", idx_a: int, idx_b: int):
        """Average two similar experts into idx_a, reinitialise idx_b."""
        for pa, pb in zip(moe_layer.experts[idx_a].parameters(),
                          moe_layer.experts[idx_b].parameters()):
            pa.data.copy_((pa.data + pb.data) / 2)
            nn.init.normal_(pb.data, std=0.02)
        # Merge router weights
        moe_layer.router.gate.weight.data[idx_a] = (
            moe_layer.router.gate.weight.data[idx_a] +
            moe_layer.router.gate.weight.data[idx_b]
        ) / 2

    @torch.no_grad()
    def step(self, moe_layer: "MoELayer", metrics: ExpertMetrics) -> dict:
        """
        Run one growth step: check metrics and apply split/prune/merge.

        Returns dict of actions taken.
        """
        actions = {"splits": [], "prunes": [], "merges": []}
        load = metrics.expert_load()
        n = len(load)

        # ── PRUNE dead experts ──────────────────────────────────────
        dead = (load < self.prune_threshold).nonzero(as_tuple=True)[0]
        for idx in dead.tolist():
            self.prune_expert(moe_layer, idx)
            actions["prunes"].append(idx)

        # ── SPLIT overloaded experts ────────────────────────────────
        overloaded = (load > self.split_threshold).nonzero(as_tuple=True)[0]
        pruned_set = set(actions["prunes"])
        for src_idx in overloaded.tolist():
            # Find a dead/pruned expert to split into
            for tgt in pruned_set:
                self.split_expert(moe_layer, src_idx, tgt)
                actions["splits"].append((src_idx, tgt))
                pruned_set.discard(tgt)
                break

        # ── MERGE similar experts ───────────────────────────────────
        sim = metrics.expert_similarity(moe_layer)
        mask = ~torch.eye(n, dtype=torch.bool)
        sim_masked = sim * mask.float()
        if sim_masked.max() > self.merge_threshold:
            idx = sim_masked.argmax()
            a, b = idx // n, idx % n
            self.merge_experts(moe_layer, a.item(), b.item())
            actions["merges"].append((a.item(), b.item()))

        return actions

