"""
Predefined model configurations for Mytho.

Usage:
    from model_configs import MODEL_CONFIGS
    cfg = MODEL_CONFIGS["100M"]
"""

MODEL_CONFIGS = {
    "10M": dict(
        d_model=128, n_heads=4, d_head=32,
        d_latent_kv=64, d_rope=16,
        n_experts=4, n_active_experts=2, d_expert_ff=384,
        max_depth=4, n_unique_blocks=1,
        seq_len=512, batch_size=16, grad_accum=2,
    ),
    "50M": dict(
        d_model=384, n_heads=6, d_head=64,
        d_latent_kv=96, d_rope=32,
        n_experts=4, n_active_experts=2, d_expert_ff=1024,
        max_depth=6, n_unique_blocks=1,
        seq_len=1024, batch_size=8, grad_accum=4,
    ),
    "100M": dict(
        d_model=512, n_heads=8, d_head=64,
        d_latent_kv=128, d_rope=32,
        n_experts=4, n_active_experts=2, d_expert_ff=1536,
        max_depth=8, n_unique_blocks=1,
        seq_len=1024, batch_size=4, grad_accum=8,
    ),
    "150M": dict(
        d_model=640, n_heads=10, d_head=64,
        d_latent_kv=192, d_rope=32,
        n_experts=6, n_active_experts=2, d_expert_ff=1792,
        max_depth=8, n_unique_blocks=1,
        seq_len=1024, batch_size=4, grad_accum=8,
    ),
    "500M": dict(
        d_model=1024, n_heads=16, d_head=64,
        d_latent_kv=384, d_rope=32,
        n_experts=8, n_active_experts=2, d_expert_ff=3072,
        max_depth=10, n_unique_blocks=2,
        seq_len=1024, batch_size=2, grad_accum=16,
    ),
    "1B": dict(
        d_model=2048, n_heads=16, d_head=128,
        d_latent_kv=512, d_rope=64,
        n_experts=12, n_active_experts=2, d_expert_ff=4096,
        max_depth=12, n_unique_blocks=3,
        seq_len=512, batch_size=1, grad_accum=32,
    ),
    "3B": dict(
        d_model=2048, n_heads=16, d_head=128,
        d_latent_kv=512, d_rope=64,
        n_experts=16, n_active_experts=2, d_expert_ff=5120,
        max_depth=16, n_unique_blocks=6,
        seq_len=128, batch_size=1, grad_accum=32,
    ),
    "7B": dict(
        d_model=4096, n_heads=32, d_head=128,
        d_latent_kv=1024, d_rope=64,
        n_experts=16, n_active_experts=2, d_expert_ff=8192,
        max_depth=24, n_unique_blocks=8,
        seq_len=128, batch_size=1, grad_accum=64,
    ),
}

VALID_SIZES = list(MODEL_CONFIGS.keys())
