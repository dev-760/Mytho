"""
Export Mytho checkpoints to Safetensors and/or ONNX format.

Usage:
    # Save weights as safetensors
    python export.py --checkpoint checkpoints_t4/step_2000.pt --format safetensors

    # Export to ONNX (fixed depth, inference-only)
    python export.py --checkpoint checkpoints_t4/step_2000.pt --format onnx

    # Both
    python export.py --checkpoint checkpoints_t4/step_2000.pt --format all

    # Custom output directory
    python export.py --checkpoint step_2000.pt --format all --output_dir exports/
"""

import argparse
import json
import os
from pathlib import Path

import torch

from mytho_model import MythoModel


def load_checkpoint(ckpt_path: str) -> tuple:
    """Load a Mytho checkpoint, return (model, config, metadata)."""
    print(f"▸ Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt["config"]
    model = MythoModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "step": ckpt.get("step", 0),
        "tokens_seen": ckpt.get("tokens_seen", 0),
    }
    n_params = model.num_parameters()
    print(f"▸ Model: {n_params:,} params ({n_params / 1e6:.1f}M)")
    print(f"▸ Step: {meta['step']}, Tokens: {meta['tokens_seen']:,}")
    return model, config, meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Safetensors export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def export_safetensors(model: MythoModel, config, meta: dict, output_dir: Path):
    """Save model weights in safetensors format."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect state dict
    state_dict = model.state_dict()

    # safetensors requires all tensors to be contiguous
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k] = v.contiguous()

    # Metadata (safetensors stores metadata as string key-value pairs)
    sf_meta = {
        "format": "mytho",
        "step": str(meta.get("step", 0)),
        "tokens_seen": str(meta.get("tokens_seen", 0)),
        "d_model": str(config.d_model),
        "n_heads": str(config.n_heads),
        "d_head": str(config.d_head),
        "n_experts": str(config.n_experts),
        "max_depth": str(config.max_depth),
        "vocab_size": str(config.vocab_size),
        "n_unique_blocks": str(config.n_unique_blocks),
    }

    # Save weights
    weights_path = output_dir / "model.safetensors"
    save_file(clean_state, str(weights_path), metadata=sf_meta)
    size_mb = weights_path.stat().st_size / 1e6
    print(f"  ✓ Safetensors weights → {weights_path} ({size_mb:.1f} MB)")

    # Save config separately as JSON
    config_path = output_dir / "config.json"
    config_dict = vars(config) if hasattr(config, "__dict__") else {}
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"  ✓ Config → {config_path}")

    return weights_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ONNX export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MythoONNXWrapper(torch.nn.Module):
    """
    Thin wrapper that runs the model in inference mode and returns only logits.

    ONNX doesn't support dict outputs or dynamic ACT loops well, so this
    wrapper runs the full forward pass with a fixed max_depth and extracts
    just the logits tensor.
    """

    def __init__(self, model: MythoModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids)
        return out["logits"]


def export_onnx(model: MythoModel, config, meta: dict, output_dir: Path,
                seq_len: int = 128, opset: int = 17):
    """Export model to ONNX format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = MythoONNXWrapper(model)
    wrapper.eval()

    # Dummy input
    dummy_input = torch.randint(1, config.vocab_size, (1, seq_len))

    onnx_path = output_dir / "model.onnx"

    print(f"  Exporting ONNX (opset={opset}, seq_len={seq_len})...")
    print(f"  Note: ACT runs to max_depth={config.max_depth} (fixed for ONNX)")

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        opset_version=opset,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        do_constant_folding=True,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  ✓ ONNX model → {onnx_path} ({size_mb:.1f} MB)")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX validation passed")
    except ImportError:
        print(f"  ⚠ Install 'onnx' package to validate: pip install onnx")
    except Exception as e:
        print(f"  ⚠ ONNX validation warning: {e}")

    return onnx_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    p = argparse.ArgumentParser(description="Export Mytho checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--format", type=str, default="all",
                   choices=["safetensors", "onnx", "all"],
                   help="Export format (default: all)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: exports/<step>/)")
    p.add_argument("--seq_len", type=int, default=128,
                   help="Sequence length for ONNX dummy input")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version")
    args = p.parse_args()

    # Load
    model, config, meta = load_checkpoint(args.checkpoint)

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        step = meta.get("step", 0)
        output_dir = Path("exports") / f"step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"▸ Output: {output_dir}")
    print("─" * 50)

    # Export safetensors
    if args.format in ("safetensors", "all"):
        print("\n[Safetensors]")
        export_safetensors(model, config, meta, output_dir)

    # Export ONNX
    if args.format in ("onnx", "all"):
        print("\n[ONNX]")
        export_onnx(model, config, meta, output_dir,
                    seq_len=args.seq_len, opset=args.opset)

    print("\n" + "─" * 50)
    print(f"▸ Export complete → {output_dir}")


if __name__ == "__main__":
    main()
