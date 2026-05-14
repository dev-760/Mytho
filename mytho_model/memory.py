"""
Hierarchical Memory System — inspired by MemGPT (Packer et al., 2023).

Implements an OS-like memory management layer for LLMs:
  • **Working Memory**: fixed-size buffer of recent/important hidden states
    that fits within the model's context window.
  • **Long-Term Memory**: unbounded archive of evicted states, indexed for
    cosine-similarity retrieval.
  • **Memory Manager**: orchestrates read/write/evict/retrieve operations
    between the two tiers.

Reference: https://arxiv.org/abs/2310.08560
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythoConfig


class WorkingMemory(nn.Module):
    """
    Fixed-capacity buffer holding the most relevant hidden states.

    When the buffer overflows, a learned *importance scorer* decides which
    entries to evict to long-term memory.
    """

    def __init__(self, capacity: int, d_model: int):
        super().__init__()
        self.capacity = capacity
        self.d_model = d_model

        # Learned importance scoring
        self.importance_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
        )

    def score_importance(self, states: torch.Tensor) -> torch.Tensor:
        """Return per-entry importance scores  [B, N, 1]."""
        return torch.sigmoid(self.importance_proj(states))

    def select_eviction_candidates(
        self, states: torch.Tensor, n_evict: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pick the *n_evict* least-important entries to evict.

        Returns:
            keep_states:   [B, capacity, D]
            evict_states:  [B, n_evict, D]
            evict_scores:  [B, n_evict, 1]
        """
        scores = self.score_importance(states).squeeze(-1)        # [B, N]
        _, keep_idx = torch.topk(scores, self.capacity, dim=-1)   # [B, cap]
        keep_idx_sorted = keep_idx.sort(dim=-1).values

        all_idx = torch.arange(states.shape[1], device=states.device)
        all_idx = all_idx.unsqueeze(0).expand(states.shape[0], -1)

        # Eviction = complement of keep set
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(1, keep_idx_sorted, False)

        keep = states.gather(1, keep_idx_sorted.unsqueeze(-1).expand(-1, -1, self.d_model))
        evict = states[mask].view(states.shape[0], n_evict, self.d_model)
        evict_scores = scores[mask].view(states.shape[0], n_evict, 1)

        return keep, evict, evict_scores


class LongTermMemory(nn.Module):
    """
    Unbounded archive with cosine-similarity retrieval.

    Stored states are L2-normalised for efficient inner-product search.
    """

    def __init__(self, d_model: int, max_entries: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.max_entries = max_entries

        # Projection for retrieval queries
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

    def retrieve(
        self,
        query: torch.Tensor,
        archive: torch.Tensor,
        top_k: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the top-k most relevant entries from the archive.

        Args:
            query:   [B, Sq, D]  current working context
            archive: [B, Na, D]  long-term memory entries

        Returns:
            retrieved: [B, top_k, D]
            scores:    [B, top_k]
        """
        if archive is None or archive.shape[1] == 0:
            empty = torch.zeros(
                query.shape[0], 0, self.d_model, device=query.device
            )
            return empty, torch.zeros(query.shape[0], 0, device=query.device)

        q = F.normalize(self.query_proj(query.mean(dim=1, keepdim=True)), dim=-1)
        k = F.normalize(self.key_proj(archive), dim=-1)

        sim = torch.matmul(q, k.transpose(-2, -1)).squeeze(1)   # [B, Na]
        actual_k = min(top_k, archive.shape[1])
        scores, idx = torch.topk(sim, actual_k, dim=-1)         # [B, k]

        retrieved = archive.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        return retrieved, scores


class MemoryManager(nn.Module):
    """
    Orchestrates the two-tier memory hierarchy.

    Pipeline per forward step:
      1. Concatenate current input with working memory.
      2. If total exceeds capacity → evict low-importance entries to LTM.
      3. Retrieve relevant entries from LTM and inject as context.
      4. Return augmented hidden states.
    """

    def __init__(self, config: MythoConfig):
        super().__init__()
        self.d_model = config.d_model
        mem_cap = config.max_seq_len // 4   # 25 % of context for memory

        self.working = WorkingMemory(capacity=mem_cap, d_model=config.d_model)
        self.ltm = LongTermMemory(config.d_model)

        # Gate to blend retrieved memory with current states
        self.mem_gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Sigmoid(),
        )

        # Per-instance state (non-persistent)
        self._wm_buffer: torch.Tensor | None = None
        self._ltm_archive: torch.Tensor | None = None

    def reset(self):
        """Clear both memory tiers (call between sequences)."""
        self._wm_buffer = None
        self._ltm_archive = None

    def forward(
        self, x: torch.Tensor, retrieve_k: int = 8
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]  current hidden states

        Returns:
            augmented: [B, S, D]  memory-augmented hidden states
        """
        B, S, D = x.shape
        device = x.device

        # ── 1. Merge with working memory ────────────────────────────
        if self._wm_buffer is not None:
            combined = torch.cat([self._wm_buffer.to(device), x], dim=1)
        else:
            combined = x

        # ── 2. Evict if over capacity ───────────────────────────────
        if combined.shape[1] > self.working.capacity:
            n_evict = combined.shape[1] - self.working.capacity
            keep, evict, _ = self.working.select_eviction_candidates(
                combined, n_evict
            )
            # Archive evicted states
            if self._ltm_archive is not None:
                self._ltm_archive = torch.cat(
                    [self._ltm_archive.to(device), evict], dim=1
                )
                # Trim if archive too large
                if self._ltm_archive.shape[1] > self.ltm.max_entries:
                    self._ltm_archive = self._ltm_archive[:, -self.ltm.max_entries:]
            else:
                self._ltm_archive = evict
            self._wm_buffer = keep.detach()
        else:
            self._wm_buffer = combined.detach()

        # ── 3. Retrieve from LTM ───────────────────────────────────
        retrieved, _ = self.ltm.retrieve(
            x, self._ltm_archive, top_k=retrieve_k
        )

        # ── 4. Gate and blend ───────────────────────────────────────
        if retrieved.shape[1] > 0:
            # Pool retrieved into a summary vector and expand
            mem_summary = retrieved.mean(dim=1, keepdim=True).expand_as(x)
            gate = self.mem_gate(torch.cat([x, mem_summary], dim=-1))
            x = x * gate + mem_summary * (1 - gate)

        return x

