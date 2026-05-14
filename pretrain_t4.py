"""
Pretrain Mytho on FineWeb-Edu — T4 GPU optimised (single-GPU, no FSDP).

Designed for Google Colab's free T4 (16 GB VRAM):
  • AMP (FP16) mixed-precision — T4 doesn't support BF16
  • Gradient accumulation for larger effective batch
  • Optional gradient checkpointing to save VRAM
  • Cosine LR schedule with linear warmup
  • Checkpointing to /content/drive or local dir
  • Live loss logging

Usage (Colab cell):
    !python pretrain_t4.py --max_steps 2000 --batch_size 4 --seq_len 512

Quick smoke-test:
    !python pretrain_t4.py --max_steps 20 --batch_size 2 --seq_len 128 --max_docs 50
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
from torch.cuda.amp import GradScaler, autocast

from mytho_model import MythoConfig, MythoModel
from mytho_model.model import MythoBlock
from data import create_dataloader, VOCAB_SIZE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cosine schedule with linear warmup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_lr(step: int, warmup: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Gradient checkpointing (trades compute for VRAM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enable_gradient_checkpointing(model: MythoModel):
    """Wrap forward of each MythoBlock with gradient checkpointing."""
    from torch.utils.checkpoint import checkpoint

    for block in model.blocks:
        original_forward = block.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(*args, **kwargs):
                # checkpoint requires at least one tensor with requires_grad
                def run(*a):
                    return orig_fn(*a, **kwargs)
                return checkpoint(run, *args, use_reentrant=False)
            return ckpt_forward

        block.forward = make_ckpt_forward(original_forward)
    print("  ✓ Gradient checkpointing enabled")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VRAM usage helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main pretraining loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pretrain(args):
    print("=" * 64)
    print("  Mytho — T4 Pretraining on FineWeb-Edu")
    print("  AMP (FP16) + AdamW | Single GPU")
    print("=" * 64)

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▸ Device: {device}")
    if torch.cuda.is_available():
        print(f"▸ GPU:    {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"▸ VRAM:   {vram:.1f} GB")

    # ── Model config (T4-optimised) ─────────────────────────────────
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
    )
    print(f"▸ Config: d={config.d_model}, h={config.n_heads}, "
          f"depth={config.max_depth}, experts={config.n_experts}, "
          f"seq={config.max_seq_len}")

    # ── Build model ─────────────────────────────────────────────────
    model = MythoModel(
        config,
        use_scratchpad=args.use_scratchpad,
        d_scratch=args.d_scratch,
    ).to(device)
    n_params = model.num_parameters()
    print(f"▸ Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ── Gradient checkpointing (optional, saves ~40% VRAM) ──────────
    if args.grad_checkpoint:
        try:
            enable_gradient_checkpointing(model)
        except Exception as e:
            print(f"  ⚠ Gradient checkpointing skipped: {e}")

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # ── AMP scaler (FP16 for T4) ────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    print(f"▸ Mixed precision: {'FP16 (AMP)' if use_amp else 'disabled'}")

    # ── Data ────────────────────────────────────────────────────────
    print(f"▸ Dataset: FineWeb-Edu ({args.subset})")
    dataloader = create_dataloader(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        subset=args.subset,
        max_docs=args.max_docs,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Checkpoint dir ──────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {k: v for k, v in vars(config).items()
                   if not k.startswith('_') and not callable(v)}
    try:
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    except Exception:
        pass

    # ── Resume ──────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        print(f"▸ Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"▸ Resumed at step {start_step}")

    # ── Training setup ──────────────────────────────────────────────
    grad_accum = args.grad_accum
    warmup = args.warmup_steps
    max_steps = args.max_steps
    min_lr = args.lr * 0.1
    eff_batch = args.batch_size * grad_accum

    print(f"▸ Batch: {args.batch_size} × {grad_accum} accum = {eff_batch} effective")
    print(f"▸ Schedule: {warmup} warmup → {max_steps} total steps")
    print("─" * 64)

    # ── Metrics tracking ────────────────────────────────────────────
    history = {
        "step": [], "loss": [], "ce_loss": [], "act_loss": [],
        "moe_loss": [], "mean_depth": [], "lr": [], "gpu_mem": [],
        "tokens_per_sec": [],
    }
    log_file = open(ckpt_dir / "train_log.jsonl", "a")

    # ── Training loop ───────────────────────────────────────────────
    model.train()
    global_step = start_step
    tokens_seen = start_step * eff_batch * args.seq_len
    t_start = time.time()
    running_loss = 0.0
    running_ce = 0.0
    data_iter = iter(dataloader)

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_ce = 0.0
        batch_depth = 0.0

        for accum_idx in range(grad_accum):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                print("  ↻ Dataset exhausted, restarting stream...")
                data_iter = iter(dataloader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()

            batch_loss += out["loss"].item() / grad_accum
            batch_ce += out["ce_loss"].item() / grad_accum
            batch_depth += out["mean_depth"].item() / grad_accum
            tokens_seen += input_ids.numel()

        # LR schedule
        lr = get_lr(global_step, warmup, max_steps, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        global_step += 1

        running_loss += batch_loss
        running_ce += batch_ce

        # ── Logging ─────────────────────────────────────────────────
        if global_step % args.log_every == 0:
            elapsed = time.time() - t_start
            avg_loss = running_loss / args.log_every
            avg_ce = running_ce / args.log_every
            tps = tokens_seen / max(elapsed, 1)
            mem = gpu_mem_gb()

            metrics = {
                "step": global_step, "loss": round(avg_loss, 4),
                "ce_loss": round(avg_ce, 4),
                "act_loss": round(out["act_loss"].item(), 6),
                "moe_loss": round(out["moe_loss"].item(), 6),
                "mean_depth": round(batch_depth, 2),
                "lr": lr, "tokens_per_sec": round(tps),
                "gpu_mem_gb": round(mem, 2),
                "timestamp": datetime.now().isoformat(),
            }
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()

            # Track for plotting
            for k in history:
                if k in metrics:
                    history[k].append(metrics[k])

            pct = global_step / max_steps * 100
            print(
                f"  [{pct:5.1f}%] Step {global_step:>6d}/{max_steps} │ "
                f"Loss {avg_loss:.4f} │ CE {avg_ce:.4f} │ "
                f"LR {lr:.2e} │ Depth {batch_depth:.1f} │ "
                f"Tok/s {tps:,.0f} │ VRAM {mem:.1f}GB"
            )
            running_loss = 0.0
            running_ce = 0.0

        # ── Checkpoint ──────────────────────────────────────────────
        if global_step % args.save_every == 0:
            path = ckpt_dir / f"step_{global_step}.pt"
            torch.save({
                "step": global_step,
                "tokens_seen": tokens_seen,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
            }, path)
            print(f"  ✓ Checkpoint → {path}")

    # ── Final save ──────────────────────────────────────────────────
    final_path = ckpt_dir / f"step_{global_step}.pt"
    torch.save({
        "step": global_step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "history": history,
    }, final_path)

    # Save history for plotting
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - t_start
    print("─" * 64)
    print(f"▸ Training complete: {global_step} steps, "
          f"{tokens_seen:,} tokens, {elapsed / 60:.1f} min")
    print(f"▸ Final checkpoint: {final_path}")
    log_file.close()

    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrain Mytho on FineWeb-Edu (T4 optimised)"
    )

    # Model — T4 defaults
    g = p.add_argument_group("Model")
    g.add_argument("--d_model",          type=int,   default=768)
    g.add_argument("--n_heads",          type=int,   default=12)
    g.add_argument("--d_head",           type=int,   default=64)
    g.add_argument("--d_latent_kv",      type=int,   default=256)
    g.add_argument("--d_rope",           type=int,   default=32)
    g.add_argument("--n_experts",        type=int,   default=8)
    g.add_argument("--n_active_experts", type=int,   default=2)
    g.add_argument("--d_expert_ff",      type=int,   default=2048)
    g.add_argument("--max_depth",        type=int,   default=8,
                   help="Reduced from 12 for T4 VRAM")
    g.add_argument("--seq_len",          type=int,   default=512,
                   help="Reduced from 2048 for T4 VRAM")
    g.add_argument("--dropout",          type=float, default=0.0)
    g.add_argument("--use_scratchpad",   action="store_true", default=False)
    g.add_argument("--d_scratch",        type=int,   default=64)

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--subset",           type=str,   default="sample-10BT")
    g.add_argument("--max_docs",         type=int,   default=None)
    g.add_argument("--num_workers",      type=int,   default=2)
    g.add_argument("--seed",             type=int,   default=42)

    # Training — T4 defaults
    g = p.add_argument_group("Training")
    g.add_argument("--batch_size",       type=int,   default=4)
    g.add_argument("--grad_accum",       type=int,   default=8,
                   help="Effective batch = batch_size × grad_accum")
    g.add_argument("--max_steps",        type=int,   default=5000)
    g.add_argument("--warmup_steps",     type=int,   default=200)
    g.add_argument("--lr",               type=float, default=3e-4)
    g.add_argument("--beta1",            type=float, default=0.9)
    g.add_argument("--beta2",            type=float, default=0.95)
    g.add_argument("--weight_decay",     type=float, default=0.1)
    g.add_argument("--max_grad_norm",    type=float, default=1.0)
    g.add_argument("--grad_checkpoint",  action="store_true", default=False,
                   help="Enable gradient checkpointing (saves VRAM)")

    # Logging
    g = p.add_argument_group("Logging")
    g.add_argument("--log_every",        type=int,   default=10)
    g.add_argument("--save_every",       type=int,   default=500)
    g.add_argument("--ckpt_dir",         type=str,   default="checkpoints_t4")
    g.add_argument("--resume",           type=str,   default=None)

    return p.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
