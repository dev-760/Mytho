"""
Text generation script for the Mytho model.

Usage:
    python generate.py --checkpoint checkpoints/mytho_epoch10.pt --prompt "Once upon"
    python generate.py --prompt "Hello world"   # random weights demo
"""

import argparse
import torch
from mytho_model import MythoModel, MythoConfig
from data import tokenise, decode, VOCAB_SIZE


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
        config = MythoConfig(vocab_size=VOCAB_SIZE)
        model = MythoModel(config).to(device)
        print("▸ Using randomly initialised model (no checkpoint)")

    model.eval()
    print(f"▸ Parameters: {model.num_parameters():,}")

    # ── Tokenise prompt with GPT-2 BPE (tiktoken) ──────────────────
    prompt_ids = tokenise(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n▸ Prompt: {args.prompt!r}")
    print(f"▸ Prompt tokens: {len(prompt_ids)}")
    print(f"▸ Generating {args.max_tokens} tokens (temperature={args.temperature})...\n")

    # ── Generate ────────────────────────────────────────────────────
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode with tiktoken
    generated_ids = output_ids[0].tolist()
    text = decode(generated_ids)
    print(f"Generated text:\n{text}")
    print(f"\n▸ Total tokens: {len(generated_ids)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate text with Mytho")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    main(p.parse_args())
