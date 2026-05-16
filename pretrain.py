"""
Pretrain Mytho on FineWeb-Edu with FSDP + AdamW  (single-GPU optimised).

FSDP on a single GPU provides:
  • CPU offloading of optimizer states → fits larger models in VRAM
  • Activation checkpointing → trades compute for memory
  • Native mixed-precision (bf16 / fp16) with loss scaling

Usage:
  # Pretrain with a named config (10M, 50M, 100M, 150M, 500M, 1B, 3B, 7B)
  python pretrain.py --model_size 100M
  python pretrain.py --model_size 500M

  # Quick smoke-test
  python pretrain.py --model_size 10M --max_docs 100 --max_steps 50

  # Custom overrides (flags override model_size defaults)
  python pretrain.py --model_size 100M --seq_len 512 --batch_size 2

  # Resume from checkpoint
  python pretrain.py --model_size 100M --resume checkpoints_pretrain/step_1000.pt
"""

import argparse
import math
import os
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import functools

from mytho_model import MythoConfig, MythoModel
from mytho_model.model import MythoBlock
from data import create_dataloader, VOCAB_SIZE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cosine schedule with linear warmup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_lr(step: int, warmup: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    # Linear warmup
    if step < warmup:
        return max_lr * (step + 1) / warmup
    # Cosine decay
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FSDP setup (single or multi-GPU via torchrun)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_fsdp():
    """Initialise distributed environment for FSDP (works with torchrun)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank


def wrap_model_fsdp(model: nn.Module, args) -> FSDP:
    """Wrap model with FSDP, configured for memory-efficient training."""

    # Auto-wrap each MythoBlock as a separate FSDP unit
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MythoBlock},
    )

    # Mixed precision policy
    if args.dtype == "bf16" and torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif args.dtype == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None

    # CPU offload for optimizer states (saves ~2× VRAM for optimizer)
    cpu_offload = CPUOffload(offload_params=args.cpu_offload)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Activation checkpointing (trades compute for memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_activation_checkpointing(model: FSDP):
    """Enable gradient checkpointing on every MythoBlock."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing as _apply_ac,
    )

    def check_fn(module): return isinstance(module, MythoBlock)
    _apply_ac(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TrainLogger:
    """Simple logger that writes to console + JSON lines file."""

    def __init__(self, log_dir: str, use_wandb: bool = False, project: str = "mytho-pretrain"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "train_log.jsonl", "a")
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=project, config={})

    def log(self, step: int, metrics: dict):
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()
        self.log_file.write(json.dumps(metrics) + "\n")
        self.log_file.flush()

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def close(self):
        self.log_file.close()
        if self.use_wandb:
            import wandb
            wandb.finish()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pretraining loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pretrain(args):
    # ── Device ──────────────────────────────────────────────────────
    local_rank = setup_fsdp()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("=" * 60)
        print("  Mytho Pretraining on FineWeb-Edu")
        print(f"  FSDP + AdamW | {world_size} GPU(s)")
        print("=" * 60)
        print(f"▸ Device: {device}")
        if torch.cuda.is_available():
            for i in range(world_size):
                print(f"▸ GPU {i}: {torch.cuda.get_device_name(i)}  "
                      f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")

    # ── Model config ────────────────────────────────────────────────
    config = MythoConfig(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_head,
        d_latent_kv=args.d_latent_kv,
        d_rope=args.d_rope,
        n_experts=args.n_experts,
        n_active_experts=args.n_active_experts,
        d_expert_ff=args.d_expert_ff,
        max_depth=args.max_depth,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        n_unique_blocks=args.n_unique_blocks,
    )
    if rank == 0:
        print(f"▸ Config: d_model={config.d_model}, heads={config.n_heads}, "
              f"depth={config.max_depth}, experts={config.n_experts}, "
              f"seq_len={config.max_seq_len}")

    # ── Build model ─────────────────────────────────────────────────
    model = MythoModel(config)
    n_params = model.num_parameters()
    if rank == 0:
        print(f"▸ Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Wrap with FSDP ──────────────────────────────────────────────
    model = wrap_model_fsdp(model, args)
    if args.activation_checkpointing:
        try:
            apply_activation_checkpointing(model)
            print("▸ Activation checkpointing: enabled")
        except Exception as e:
            print(f"▸ Activation checkpointing: skipped ({e})")

    # ── Optimizer (AdamW) ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),   # fused AdamW if on CUDA
    )
    if rank == 0:
        print(f"▸ Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")

    # ── Data ────────────────────────────────────────────────────────
    if rank == 0:
        print(f"▸ Dataset: FineWeb-Edu ({args.subset})")
    dataloader = create_dataloader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        subset=args.subset,
        max_docs=args.max_docs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Logger ──────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(
        args.ckpt_dir, use_wandb=args.wandb) if rank == 0 else None

    # ── Resume from checkpoint ──────────────────────────────────────
    start_step = 0
    if args.resume:
        if rank == 0:
            print(f"▸ Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # FSDP full_state_dict loading
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        if rank == 0:
            print(f"▸ Resumed at step {start_step}")

    # ── Save config ─────────────────────────────────────────────────
    if rank == 0:
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(vars(config) if hasattr(config, '__dict__')
                      else str(config), f, indent=2, default=str)

    # ── Training ────────────────────────────────────────────────────
    grad_accum_steps = args.grad_accum
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    min_lr = args.lr * 0.1

    if rank == 0:
        print(f"▸ Batch: {args.batch_size} × {grad_accum_steps} accum "
              f"= {args.batch_size * grad_accum_steps} effective")
        print(f"▸ Schedule: {warmup_steps} warmup → {max_steps} total steps")
        print(f"▸ Precision: {args.dtype}")
        print("─" * 60)

    model.train()
    global_step = start_step
    tokens_seen = start_step * args.batch_size * args.seq_len * grad_accum_steps
    t_start = time.time()
    running_loss = 0.0
    running_ce = 0.0
    micro_step = 0

    data_iter = iter(dataloader)

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_ce = 0.0
        batch_depth = 0.0

        for accum_idx in range(grad_accum_steps):
            # Get batch
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                print("▸ Dataset exhausted, restarting stream...")
                data_iter = iter(dataloader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward
            out = model(input_ids, labels=labels)
            loss = out["loss"] / grad_accum_steps

            # Backward
            loss.backward()

            batch_loss += out["loss"].item() / grad_accum_steps
            batch_ce += out["ce_loss"].item() / grad_accum_steps
            batch_depth += out["mean_depth"].item() / grad_accum_steps
            micro_step += 1
            tokens_seen += input_ids.numel()

        # ── LR schedule ─────────────────────────────────────────────
        lr = get_lr(global_step, warmup_steps, max_steps, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient clipping ───────────────────────────────────────
        if args.max_grad_norm > 0:
            model.clip_grad_norm_(args.max_grad_norm)

        # ── Optimizer step ──────────────────────────────────────────
        optimizer.step()
        global_step += 1

        running_loss += batch_loss
        running_ce += batch_ce

        # ── Logging ─────────────────────────────────────────────────
        if global_step % args.log_every == 0:
            elapsed = time.time() - t_start
            avg_loss = running_loss / args.log_every
            avg_ce = running_ce / args.log_every
            tps = tokens_seen / elapsed
            gpu_mem = (torch.cuda.max_memory_allocated() / 1e9
                       if torch.cuda.is_available() else 0)

            metrics = {
                "loss": round(avg_loss, 4),
                "ce_loss": round(avg_ce, 4),
                "act_loss": round(out["act_loss"].item(), 6),
                "moe_loss": round(out["moe_loss"].item(), 6),
                "mean_depth": round(batch_depth, 2),
                "lr": lr,
                "tokens": tokens_seen,
                "tokens_per_sec": round(tps, 0),
                "gpu_mem_gb": round(gpu_mem, 2),
            }
            if rank == 0:
                logger.log(global_step, metrics)
                print(
                    f"  Step {global_step:>6d}/{max_steps} │ "
                    f"Loss {avg_loss:.4f} │ CE {avg_ce:.4f} │ "
                    f"LR {lr:.2e} │ Depth {batch_depth:.1f} │ "
                    f"Tok/s {tps:,.0f} │ GPU {gpu_mem:.1f}GB"
                )
            running_loss = 0.0
            running_ce = 0.0

            # ── Checkpoint ──────────────────────────────────────────────
            if global_step % args.save_every == 0 and rank == 0:
                save_checkpoint(model, optimizer, config, global_step,
                                tokens_seen, ckpt_dir, args.state_dict_type)

    # ── Final checkpoint ────────────────────────────────────────────
    if rank == 0:
        save_checkpoint(model, optimizer, config, global_step, tokens_seen,
                        ckpt_dir, args.state_dict_type)
    elapsed = time.time() - t_start
    if rank == 0:
        print("─" * 60)
        print(f"▸ Training complete: {global_step} steps, "
              f"{tokens_seen:,} tokens, {elapsed / 3600:.1f}h")
        logger.close()
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, config, step, tokens, ckpt_dir, state_dict_type: str = "full"):
    """Save FSDP checkpoint (.pt, plus optional .safetensors for full state)."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    if state_dict_type == "sharded":
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_sd = model.state_dict()
    else:
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model_sd = model.state_dict()

    # PyTorch checkpoint (weights + optimizer + config)
    state = {
        "step": step,
        "tokens_seen": tokens,
        "model_state_dict": model_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    pt_path = Path(ckpt_dir) / f"step_{step}.pt"
    torch.save(state, pt_path)
    print(f"  ✓ Checkpoint → {pt_path}")

    # Safetensors (weights only, portable) only for FULL state dicts
    if state_dict_type == "full":
        try:
            from safetensors.torch import save_file
            sf_path = Path(ckpt_dir) / f"step_{step}.safetensors"
            clean = {k: v.contiguous() for k, v in model_sd.items()}
            save_file(clean, str(sf_path), metadata={
                "step": str(step), "tokens_seen": str(tokens),
                "format": "mytho",
            })
            print(f"  ✓ Safetensors → {sf_path}")
        except ImportError:
            pass  # safetensors not installed, skip silently
    else:
        print("  Note: Skipping safetensors for sharded state dict.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    p = argparse.ArgumentParser(description="Pretrain Mytho on FineWeb-Edu")

    # Model size preset
    p.add_argument("--model_size",      type=str,   default=None,
                   choices=["10M", "50M", "100M",
                            "150M", "500M", "1B", "3B", "7B"],
                   help="Named model config (overrides architecture defaults)")

    # Model architecture
    g = p.add_argument_group("Model")
    g.add_argument("--d_model",         type=int,   default=768)
    g.add_argument("--n_heads",         type=int,   default=12)
    g.add_argument("--d_head",          type=int,   default=64)
    g.add_argument("--d_latent_kv",     type=int,   default=256)
    g.add_argument("--d_rope",          type=int,   default=32)
    g.add_argument("--n_experts",       type=int,   default=8)
    g.add_argument("--n_active_experts", type=int,   default=2)
    g.add_argument("--d_expert_ff",     type=int,   default=2048)
    g.add_argument("--max_depth",       type=int,   default=12)
    g.add_argument("--seq_len",         type=int,   default=1024)
    g.add_argument("--dropout",         type=float, default=0.0)
    g.add_argument("--n_unique_blocks", type=int,   default=1,
                   help="Number of distinct transformer blocks (1=weight-tied)")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--subset",          type=str,   default="sample-10BT",
                   help="FineWeb-Edu subset: sample-10BT, sample-100BT, default")
    g.add_argument("--max_docs",        type=int,   default=None,
                   help="Cap documents (for debugging)")
    g.add_argument("--num_workers",     type=int,   default=2)
    g.add_argument("--seed",            type=int,   default=42)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--batch_size",      type=int,   default=8)
    g.add_argument("--grad_accum",      type=int,   default=4,
                   help="Gradient accumulation steps")
    g.add_argument("--max_steps",       type=int,   default=50_000)
    g.add_argument("--warmup_steps",    type=int,   default=1000)
    g.add_argument("--lr",              type=float, default=3e-4)
    g.add_argument("--min_lr",          type=float, default=3e-5)
    g.add_argument("--beta1",           type=float, default=0.9)
    g.add_argument("--beta2",           type=float, default=0.95)
    g.add_argument("--eps",             type=float, default=1e-8)
    g.add_argument("--weight_decay",    type=float, default=0.1)
    g.add_argument("--max_grad_norm",   type=float, default=1.0)

    # FSDP
    g = p.add_argument_group("FSDP")
    g.add_argument("--dtype",           type=str,   default="bf16",
                   choices=["fp32", "fp16", "bf16"])
    g.add_argument("--cpu_offload",     action="store_true", default=False,
                   help="Offload FSDP params to CPU (saves VRAM)")
    g.add_argument("--activation_checkpointing",
                   action="store_true", default=True)
    g.add_argument("--state_dict_type", type=str, default="full",
                   choices=["full", "sharded"],
                   help="Checkpoint format: full (gathers weights) or sharded (faster)")

    # Logging / checkpointing
    g = p.add_argument_group("Logging")
    g.add_argument("--log_every",       type=int,   default=10)
    g.add_argument("--save_every",      type=int,   default=1000)
    g.add_argument("--ckpt_dir",        type=str,
                   default="checkpoints_pretrain")
    g.add_argument("--wandb",           action="store_true", default=False)
    g.add_argument("--resume",          type=str,   default=None,
                   help="Path to checkpoint to resume from")

    args = p.parse_args()

    # Apply model_size preset (CLI flags override preset values)
    if args.model_size:
        from model_configs import MODEL_CONFIGS
        import sys
        preset = MODEL_CONFIGS[args.model_size]
        for key, val in preset.items():
            if f"--{key}" not in " ".join(sys.argv):
                setattr(args, key, val)
        print(f"▸ Using Mytho-{args.model_size} preset")

    return args


if __name__ == "__main__":
    pretrain(parse_args())
