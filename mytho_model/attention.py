"""
Multi-Latent Attention (MLA) – inspired by DeepSeek-V2.

Key idea: compress K and V into a shared low-rank latent vector before
decompressing, dramatically reducing KV-cache size while preserving quality.

    Input ─► Q projection  ──────────────────────────► Q_content  ─┐
         ├► Q RoPE proj   ──► apply_rope ──────────► Q_rope     ─┤
         └► KV down-proj  ──► kv_norm ──► latent ──┬► K_up     ─┤
                                                    ├► V_up     ─┤
                                                    └► K_rope   ─┘
    Attention(Q=[Q_content‖Q_rope], K=[K_content‖K_rope], V)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MythoConfig
from .components import RMSNorm, apply_rotary_embedding


class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention with decoupled Rotary Position Embeddings.

    • Q is projected at full rank, split into *content* + *RoPE* heads.
    • K & V are first compressed into a low-rank latent ``c_kv``, then
      decompressed.  A separate K-RoPE projection is applied on the latent.
    • This means the KV cache only needs to store the latent vector
      (size ``d_latent_kv``) instead of full K & V per head.
    """

    def __init__(self, config: MythoConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_rope = config.d_rope
        self.d_latent_kv = config.d_latent_kv

        self.scale = 1.0 / math.sqrt(self.d_head + self.d_rope)

        # ── Query projections ───────────────────────────────────────
        self.W_q_content = nn.Linear(
            config.d_model, config.n_heads * config.d_head, bias=False
        )
        self.W_q_rope = nn.Linear(
            config.d_model, config.n_heads * config.d_rope, bias=False
        )
        self.q_norm = RMSNorm(config.d_head)

        # ── KV compression (down-projection to latent) ──────────────
        self.W_kv_down = nn.Linear(
            config.d_model, config.d_latent_kv, bias=False
        )
        self.kv_norm = RMSNorm(config.d_latent_kv)

        # ── KV decompression (up-projection from latent) ────────────
        self.W_k_content = nn.Linear(
            config.d_latent_kv, config.n_heads * config.d_head, bias=False
        )
        self.W_v = nn.Linear(
            config.d_latent_kv, config.n_heads * config.d_head, bias=False
        )
        self.W_k_rope = nn.Linear(
            config.d_latent_kv, config.n_heads * config.d_rope, bias=False
        )

        # ── Output projection ──────────────────────────────────────
        self.W_o = nn.Linear(
            config.n_heads * config.d_head, config.d_model, bias=False
        )
        self.attn_drop = nn.Dropout(config.dropout)

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        cache_step: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, S, D]
            rope_cos:  [max_seq, d_rope // 2]
            rope_sin:  [max_seq, d_rope // 2]
            mask:      [B, 1, S, S]  causal mask (True = attend)
            kv_cache:  optional dict holding {"latent": Tensor} for past tokens
            cache_step: current generation step (for cache indexing)

        Returns:
            output:    [B, S, D]
        """
        B, S, _ = x.shape
        H, Dh, Dr = self.n_heads, self.d_head, self.d_rope

        # ── Queries ─────────────────────────────────────────────────
        q_c = self.W_q_content(x).view(B, S, H, Dh).transpose(1, 2)     # [B,H,S,Dh]
        q_c = self.q_norm(q_c)
        q_r = self.W_q_rope(x).view(B, S, H, Dr).transpose(1, 2)       # [B,H,S,Dr]
        q_r = apply_rotary_embedding(q_r, rope_cos, rope_sin)

        # ── KV latent ───────────────────────────────────────────────
        c_kv = self.kv_norm(self.W_kv_down(x))                          # [B,S,d_lat]

        # Cache handling: store the compressed latent (much smaller)
        if kv_cache is not None:
            if "latent" in kv_cache and kv_cache["latent"] is not None:
                c_kv = torch.cat([kv_cache["latent"], c_kv], dim=1)
            kv_cache["latent"] = c_kv

        # ── Decompress ──────────────────────────────────────────────
        k_c = self.W_k_content(c_kv).view(B, -1, H, Dh).transpose(1, 2)  # [B,H,Sk,Dh]
        v = self.W_v(c_kv).view(B, -1, H, Dh).transpose(1, 2)            # [B,H,Sk,Dh]
        k_r = self.W_k_rope(c_kv).view(B, -1, H, Dr).transpose(1, 2)     # [B,H,Sk,Dr]
        k_r = apply_rotary_embedding(k_r, rope_cos, rope_sin)

        # ── Concatenate content + RoPE components ───────────────────
        q = torch.cat([q_c, q_r], dim=-1)   # [B, H, Sq, Dh+Dr]
        k = torch.cat([k_c, k_r], dim=-1)   # [B, H, Sk, Dh+Dr]

        # ── Scaled dot-product attention ────────────────────────────
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Sq,Sk]

        if mask is not None:
            attn = attn.masked_fill(~mask[:, :, :S, :k.shape[2]], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                        # [B, H, S, Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, H * Dh)
        return self.W_o(out)

