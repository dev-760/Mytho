"""
Deep Ensemble Uncertainty — Lakshminarayanan et al. (2017).

Estimate predictive uncertainty by running multiple stochastic forward
passes (MC-Dropout) or maintaining an ensemble of model heads.

This module provides:
  • ``MCDropoutEstimator``:  uses dropout-at-inference for cheap approximation.
  • ``EnsembleHead``:        multiple LM heads sharing the same backbone.

Reference: https://arxiv.org/abs/1612.01474
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MythoModel


class MCDropoutEstimator:
    """
    Monte-Carlo Dropout uncertainty estimation.

    Keeps dropout enabled at inference and runs *k* forward passes
    to estimate mean prediction and predictive entropy.
    """

    def __init__(self, model: "MythoModel", n_samples: int = 5):
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self):
        """Turn on all Dropout layers even in eval mode."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def estimate(self, input_ids: torch.Tensor) -> dict:
        """
        Run multiple forward passes and compute uncertainty metrics.

        Returns dict with:
            mean_logits:  [B, S, V]  averaged logits
            mean_probs:   [B, S, V]  averaged probabilities
            entropy:      [B, S]     predictive entropy (higher = more uncertain)
            variance:     [B, S]     mean token-level variance
            all_logits:   list of [B, S, V] per sample
        """
        self.model.eval()
        self._enable_dropout()

        all_logits = []
        for _ in range(self.n_samples):
            out = self.model(input_ids)
            all_logits.append(out["logits"])

        stacked = torch.stack(all_logits, dim=0)          # [K, B, S, V]
        mean_logits = stacked.mean(dim=0)                  # [B, S, V]
        mean_probs = F.softmax(stacked, dim=-1).mean(0)    # [B, S, V]

        # Predictive entropy: H = -Σ p log p
        entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(dim=-1)  # [B, S]

        # Token-level variance (avg across vocab)
        variance = stacked.var(dim=0).mean(dim=-1)          # [B, S]

        self.model.eval()  # restore full eval
        return {
            "mean_logits": mean_logits,
            "mean_probs": mean_probs,
            "entropy": entropy,
            "variance": variance,
            "all_logits": all_logits,
        }


class EnsembleHead(nn.Module):
    """
    Multiple independent LM heads sharing the same transformer backbone.

    Each head produces its own logits; disagreement between heads signals
    epistemic uncertainty.
    """

    def __init__(self, n_heads: int, d_model: int, vocab_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_heads)
        ])

    def forward(self, hidden: torch.Tensor) -> dict:
        """
        Args:
            hidden: [B, S, D]  backbone output

        Returns dict:
            mean_logits: [B, S, V]
            ensemble_logits: list of [B, S, V]
            disagreement: [B, S]  pairwise KL between heads (mean)
        """
        logits_list = [head(hidden) for head in self.heads]
        stacked = torch.stack(logits_list, dim=0)            # [K, B, S, V]
        mean_logits = stacked.mean(dim=0)

        # Pairwise disagreement via KL divergence
        probs = F.softmax(stacked, dim=-1)                    # [K, B, S, V]
        mean_probs = probs.mean(dim=0, keepdim=True)          # [1, B, S, V]
        kl = (probs * (torch.log(probs + 1e-12) - torch.log(mean_probs + 1e-12))).sum(-1)
        disagreement = kl.mean(dim=0)                          # [B, S]

        return {
            "mean_logits": mean_logits,
            "ensemble_logits": logits_list,
            "disagreement": disagreement,
        }

