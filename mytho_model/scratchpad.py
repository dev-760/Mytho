"""
Latent Scratchpad — persistent internal workspace for recurrent reasoning.

The scratchpad is a tensor [B, T, d_scratch] that persists across recurrent
depth steps, providing a compressed working memory that:
  • Attention reads from (via learned projection to d_model)
  • Experts write to (via gated update after MoE)
  • Verifier reads from (for confidence/quality assessment)

This creates internal reasoning without generating tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm


class LatentScratchpad(nn.Module):
    """
    Persistent scratchpad workspace updated at each recurrent depth step.

    Architecture:
        read:  scratch → proj → d_model context (added to attention input)
        write: gated update from expert output → scratch += delta
    """

    def __init__(self, d_model: int, d_scratch: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_scratch = d_scratch

        # ── Read pathway (scratchpad → attention context) ───────────
        self.read_norm = RMSNorm(d_scratch)
        self.read_proj = nn.Linear(d_scratch, d_model, bias=False)

        # ── Write pathway (expert output → scratchpad update) ───────
        self.write_norm = RMSNorm(d_model)
        self.write_proj = nn.Linear(d_model, d_scratch, bias=False)

        # ── Gated update (controls how much to write) ───────────────
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model + d_scratch, d_scratch),
            nn.Sigmoid(),
        )

        # ── Delta computation (what to write) ───────────────────────
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model + d_scratch, d_scratch),
            nn.SiLU(),
            nn.Linear(d_scratch, d_scratch),
        )

        # ── Cross-attention: hidden states attend to scratchpad ─────
        self.cross_attn_q = nn.Linear(d_model, n_heads * (d_model // n_heads), bias=False)
        self.cross_attn_k = nn.Linear(d_scratch, n_heads * (d_model // n_heads), bias=False)
        self.cross_attn_v = nn.Linear(d_scratch, n_heads * (d_model // n_heads), bias=False)
        self.cross_attn_o = nn.Linear(n_heads * (d_model // n_heads), d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def init_scratch(self, batch: int, seq_len: int,
                     device: torch.device) -> torch.Tensor:
        """Create a fresh zeroed scratchpad [B, T, d_scratch]."""
        return torch.zeros(batch, seq_len, self.d_scratch, device=device)

    def read(self, hidden: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
        """
        Read from scratchpad via cross-attention.

        Args:
            hidden:  [B, T, d_model]  current hidden states
            scratch: [B, T, d_scratch]

        Returns:
            context: [B, T, d_model]  scratchpad information to add to hidden
        """
        B, T, _ = hidden.shape
        H, Dh = self.n_heads, self.head_dim

        s = self.read_norm(scratch)

        q = self.cross_attn_q(hidden).view(B, T, H, Dh).transpose(1, 2)
        k = self.cross_attn_k(s).view(B, T, H, Dh).transpose(1, 2)
        v = self.cross_attn_v(s).view(B, T, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.cross_attn_o(out)

    def write(self, hidden: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
        """
        Gated update: scratch += gate * delta.

        Args:
            hidden:  [B, T, d_model]  expert output
            scratch: [B, T, d_scratch]

        Returns:
            updated_scratch: [B, T, d_scratch]
        """
        h = self.write_norm(hidden)
        combined = torch.cat([h, scratch], dim=-1)

        gate = self.gate_proj(combined)       # [B, T, d_scratch]
        delta = self.delta_proj(combined)     # [B, T, d_scratch]

        return scratch + gate * delta

