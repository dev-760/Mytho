"""
Branching Recurrence — latent beam search within the recurrent depth loop.

Instead of a single recurrent trajectory, branch into N latent paths
at specified depths. The verifier scores each branch and the best
survives (or a weighted combination is used).

This is the recurrent equivalent of self-consistency / tree reasoning /
beam search, but MUCH cheaper because all branches share:
  • weights
  • KV cache structure
  • expert parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm


class BranchingController(nn.Module):
    """
    Controls latent branching and selection within recurrent depth.

    At branch points:
      1. Perturb the hidden state to create N diverse branches.
      2. Each branch runs through the shared block independently.
      3. Verifier scores each branch.
      4. Best branch is selected (or soft combination).
    """

    def __init__(self, d_model: int, n_branches: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_branches = n_branches

        # Learned perturbation for each branch (diversity injection)
        self.branch_perturbations = nn.Parameter(
            torch.randn(n_branches, d_model) * 0.01
        )

        # Branch diversity projection (ensures branches are meaningfully different)
        self.diversity_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_branches)
        ])

        # Merge gate (for soft combination mode)
        self.merge_gate = nn.Sequential(
            nn.Linear(d_model * n_branches, n_branches),
            nn.Softmax(dim=-1),
        )

        self.branch_norm = RMSNorm(d_model)

    def branch(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Create N diverse branches from hidden state x.

        Args:
            x: [B, T, D] hidden states

        Returns:
            branches: list of N tensors, each [B, T, D]
        """
        branches = []
        for i in range(self.n_branches):
            perturbation = self.branch_perturbations[i]  # [D]
            branch_i = x + perturbation.unsqueeze(0).unsqueeze(0)
            branch_i = self.diversity_proj[i](branch_i)
            branch_i = self.branch_norm(branch_i)
            # Residual: keep most of the original, add diversity
            branch_i = x + 0.1 * branch_i
            branches.append(branch_i)
        return branches

    def select_hard(
        self,
        branches: list[torch.Tensor],
        scores: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Hard selection: pick the branch with highest verifier score.

        Args:
            branches: list of N [B, T, D] tensors
            scores:   list of N [B, T, 1] confidence scores

        Returns:
            selected: [B, T, D]
        """
        stacked_scores = torch.cat(scores, dim=-1)  # [B, T, N]
        best_idx = stacked_scores.argmax(dim=-1, keepdim=True)  # [B, T, 1]
        best_idx = best_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)

        stacked = torch.stack(branches, dim=2)  # [B, T, N, D]
        selected = stacked.gather(2, best_idx).squeeze(2)  # [B, T, D]
        return selected

    def select_soft(
        self,
        branches: list[torch.Tensor],
        scores: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Soft selection: weighted combination by verifier scores.

        Args:
            branches: list of N [B, T, D] tensors
            scores:   list of N [B, T, 1] confidence scores

        Returns:
            combined: [B, T, D]
        """
        weights = torch.cat(scores, dim=-1)  # [B, T, N]
        weights = F.softmax(weights, dim=-1)  # normalise

        stacked = torch.stack(branches, dim=2)  # [B, T, N, D]
        weights = weights.unsqueeze(-1)  # [B, T, N, 1]
        combined = (stacked * weights).sum(dim=2)  # [B, T, D]
        return combined

    def branch_disagreement(
        self, branches: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Measure disagreement between branches (for uncertainty estimation).

        Returns:
            disagreement: [B, T, 1] — higher = more uncertain
        """
        stacked = torch.stack(branches, dim=0)  # [N, B, T, D]
        variance = stacked.var(dim=0).mean(dim=-1, keepdim=True)  # [B, T, 1]
        return variance

