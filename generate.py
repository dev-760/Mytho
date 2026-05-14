"""
Text generation script for the Mytho model.

Usage:
    python generate.py --checkpoint checkpoints/mytho_epoch10.pt --prompt "Once upon"
"""

import argparse
import torch
from mytho_model import MythoModel, MythoConfig


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ─────────────────────────────────────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = MythoModel(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"▸ Loaded checkpoint from {args.checkpoint}")
    else:
        # Demo mode with random weights
        config = MythoConfig(vocab_size=args.vocab_size)
        model = MythoModel(config).to(device)
        print("▸ Using randomly initialised model (no checkpoint)")

    model.eval()
    print(f"▸ Parameters: {model.num_parameters():,}")

    # ── Tokenise prompt (simple char-level for demo) ────────────────
    # In production, replace with a proper tokeniser (e.g. SentencePiece)
    prompt_ids = [ord(c) % config.vocab_size for c in args.prompt]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n▸ Prompt: {args.prompt!r}")
    print(f"▸ Generating {args.max_tokens} tokens (temperature={args.temperature})...\n")

    # ── Generate ────────────────────────────────────────────────────
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode (char-level for demo)
    generated = output_ids[0].tolist()
    text = "".join(chr(t % 128) if 32 <= (t % 128) < 127 else "·" for t in generated)
    print(f"Generated text:\n{text}")
    print(f"\n▸ Total tokens: {len(generated)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate text with Mytho")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--vocab_size", type=int, default=32000)
    main(p.parse_args())

