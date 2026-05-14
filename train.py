"""
Training script for the Recurrent-Depth Transformer (Mytho).

Supports:
  • Learning-rate warm-up + cosine decay
  • Gradient clipping
  • Mixed-precision (AMP) training
  • Periodic checkpointing
  • Logging of CE / ACT / MoE losses and mean depth

Usage:
    python train.py                          # train with defaults
    python train.py --epochs 20 --lr 3e-4    # override hyperparams
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from mytho_model import MythoConfig, MythoModel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Synthetic data loader (replace with real tokenised data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def synthetic_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
):
    """Yields random token batches for smoke-testing."""
    for _ in range(n_batches):
        ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        yield ids, ids.clone()  # input_ids, labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Learning-rate scheduler: linear warm-up → cosine decay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▸ Device: {device}")

    # ── Model ───────────────────────────────────────────────────────
    config = MythoConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_head,
        d_latent_kv=args.d_latent_kv,
        n_experts=args.n_experts,
        n_active_experts=args.n_active_experts,
        max_depth=args.max_depth,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    model = MythoModel(config).to(device)

    n_params = model.num_parameters()
    print(f"▸ Model parameters: {n_params:,}")
    print(f"▸ Config: {config}")

    # ── Optimiser ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = int(total_steps * 0.05)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Checkpoint directory ────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Training ────────────────────────────────────────────────────
    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        loader = synthetic_dataloader(
            config.vocab_size, args.seq_len, args.batch_size,
            args.steps_per_epoch, device,
        )

        epoch_loss = 0.0
        t0 = time.time()

        for step, (input_ids, labels) in enumerate(loader, 1):
            # LR schedule
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(input_ids, labels=labels)
                loss = out["loss"]

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if step % args.log_every == 0:
                avg = epoch_loss / step
                depth = out["mean_depth"].item()
                print(
                    f"  Epoch {epoch} │ Step {step:>4d}/{args.steps_per_epoch} │ "
                    f"LR {lr:.2e} │ Loss {loss.item():.4f} (avg {avg:.4f}) │ "
                    f"CE {out['ce_loss'].item():.4f} │ "
                    f"ACT {out['act_loss'].item():.6f} │ "
                    f"MoE {out['moe_loss'].item():.6f} │ "
                    f"Depth {depth:.2f}"
                )

        elapsed = time.time() - t0
        print(
            f"═ Epoch {epoch} done │ "
            f"Avg loss {epoch_loss / args.steps_per_epoch:.4f} │ "
            f"Time {elapsed:.1f}s"
        )

        # Save checkpoint
        if epoch % args.save_every == 0:
            path = os.path.join(args.ckpt_dir, f"mytho_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                path,
            )
            print(f"  ✓ Saved checkpoint → {path}")

    print("▸ Training complete.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a Recurrent-Depth Transformer")

    # Model
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--d_latent_kv", type=int, default=256)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--n_active_experts", type=int, default=2)
    p.add_argument("--max_depth", type=int, default=12)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Misc
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")

    train(p.parse_args())

