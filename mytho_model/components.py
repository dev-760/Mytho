"""
Shared building blocks: RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RMSNorm – Pre-norm layer used throughout the model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rotary Position Embeddings (RoPE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def precompute_rope_frequencies(
    dim: int, max_seq_len: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape [max_seq_len, dim // 2]."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)          # [seq, dim//2]
    return angles.cos(), angles.sin()


def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to tensor *x* of shape [B, H, S, D].
    cos / sin have shape [max_seq, D//2]; we slice to S automatically.
    """
    seq_len = x.shape[2]
    d = x.shape[-1]
    half = d // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    cos_s = cos[:seq_len].unsqueeze(0).unsqueeze(0)   # [1, 1, S, half]
    sin_s = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos_s - x2 * sin_s
    out2 = x1 * sin_s + x2 * cos_s
    return torch.cat([out1, out2], dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SwiGLU Feed-Forward (used inside each expert)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SwiGLU(nn.Module):
    """
    SwiGLU activation from Shazeer (2020).
    gate = Swish(W_gate · x)
    out  = W_down( gate ⊙ (W_up · x) )
    """

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        return self.dropout(self.w_down(gate * self.w_up(x)))

