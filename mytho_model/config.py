"""
Configuration for the Recurrent-Depth Transformer model.
"""

from dataclasses import dataclass


@dataclass
class MythoConfig:
    """All hyperparameters for the Mytho model."""

    # ── Vocabulary ──────────────────────────────────────────────────
    vocab_size: int = 32_000
    pad_token_id: int = 0

    # ── Core Dimensions ─────────────────────────────────────────────
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64

    # ── Multi-Latent Attention (MLA) ────────────────────────────────
    d_latent_kv: int = 256          # KV compression bottleneck
    d_rope: int = 32                # RoPE dimensions per head

    # ── Mixture of Experts (MoE) ────────────────────────────────────
    n_experts: int = 8
    n_active_experts: int = 2       # top-k experts per token
    d_expert_ff: int = 2048         # hidden dim inside each expert
    expert_balance_coeff: float = 0.01  # auxiliary load-balancing weight
    use_switch_moe: bool = False        # use Switch Transformer top-1 routing
    switch_capacity_factor: float = 1.25  # capacity factor for SwitchMoE

    # ── Recurrent Depth ─────────────────────────────────────────────
    max_depth: int = 12             # maximum recurrent iterations
    n_unique_blocks: int = 1        # distinct transformer blocks (weight-tied)

    # ── Adaptive Computation Time (ACT) ─────────────────────────────
    act_threshold: float = 0.99     # cumulative halt threshold
    act_loss_coeff: float = 0.001   # ponder cost weight in loss

    # ── Sequence ────────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Regularisation ──────────────────────────────────────────────
    dropout: float = 0.1
    rope_base: float = 10_000.0
    init_std: float = 0.02

    # ── Derived helpers ─────────────────────────────────────────────
    @property
    def total_q_dim(self) -> int:
        """Full query dimension including RoPE components."""
        return self.n_heads * (self.d_head + self.d_rope)

    @property
    def total_kv_content_dim(self) -> int:
        return self.n_heads * self.d_head

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_active_experts <= self.n_experts, "active experts must be <= total experts"
        assert self.max_depth >= 1, "max_depth must be >= 1"

