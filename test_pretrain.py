"""Quick validation: test data pipeline + FSDP wrapping without network access."""
import os, sys, torch

# 1. Test tokeniser
print("1. Testing tiktoken tokeniser...")
from data import get_tokeniser, tokenise, decode, VOCAB_SIZE
tok = get_tokeniser()
text = "The recurrent depth transformer is a novel architecture."
ids = tokenise(text)
back = decode(ids)
assert back == text, f"Roundtrip failed: {back!r}"
print(f"   Vocab: {VOCAB_SIZE}  Encoded: {len(ids)} tokens  ✓")

# 2. Test model builds with correct vocab
print("2. Building model with VOCAB_SIZE...")
from mytho_model import MythoConfig, MythoModel
cfg = MythoConfig(
    vocab_size=VOCAB_SIZE, d_model=256, n_heads=4, d_head=64,
    d_latent_kv=64, d_rope=16, max_depth=4,
    n_experts=4, n_active_experts=2, d_expert_ff=512,
    max_seq_len=128,
)
model = MythoModel(cfg)
print(f"   Params: {model.num_parameters():,}  ✓")

# 3. Test forward pass with vocab-sized inputs
print("3. Forward pass...")
ids = torch.randint(1, VOCAB_SIZE, (2, 64))
out = model(ids, labels=ids)
print(f"   Loss: {out['loss'].item():.4f}  Depth: {out['mean_depth'].item():.1f}  ✓")

# 4. Test FSDP wrapping (if CUDA available)
if torch.cuda.is_available():
    print("4. Testing FSDP wrapping...")
    from pretrain import setup_fsdp, wrap_model_fsdp
    import argparse
    args = argparse.Namespace(dtype="fp32", cpu_offload=False, activation_checkpointing=False)
    setup_fsdp()
    model_fsdp = wrap_model_fsdp(MythoModel(cfg), args)
    ids_gpu = ids.cuda()
    out_fsdp = model_fsdp(ids_gpu, labels=ids_gpu)
    print(f"   FSDP Loss: {out_fsdp['loss'].item():.4f}  ✓")
    import torch.distributed as dist
    dist.destroy_process_group()
else:
    print("4. Skipping FSDP test (no CUDA)")

print("\n ALL PRETRAIN VALIDATIONS PASSED")

