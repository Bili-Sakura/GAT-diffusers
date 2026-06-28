#!/usr/bin/env python3
"""Convert a legacy GAT training checkpoint (.pt) to a Diffusers-style pipeline folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusers import GATGenerator, GAT_MODEL_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy GAT .pt checkpoint to Diffusers pipeline folder.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to legacy training checkpoint (.pt).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output Diffusers pipeline directory.")
    parser.add_argument("--model", type=str, default=None, choices=list(GAT_MODEL_PRESETS.keys()))
    parser.add_argument("--resolution", type=int, default=256, choices=[128, 256, 512])
    parser.add_argument("--weight-key", type=str, default="ema", choices=["ema", "generator", "model"])
    parser.add_argument("--save-discriminator", action="store_true")
    parser.add_argument("--truncation-psi", type=float, default=0.3)
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-depth", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    vae_hub_id = f"stabilityai/sd-vae-ft-{args.vae}"
    out = GATGenerator.convert_checkpoint(
        args.ckpt,
        args.output_dir,
        model_name=args.model,
        resolution=args.resolution,
        weight_key=args.weight_key,
        legacy=args.legacy,
        encoder_depth=args.encoder_depth,
        save_discriminator=args.save_discriminator,
        truncation_psi=args.truncation_psi,
        vae_hub_id=vae_hub_id,
    )
    print(f"Saved Diffusers pipeline to {out.resolve()}")
    print("Load with:")
    print("  from diffusers import GATPipeline")
    print(f"  pipe = GATPipeline.from_pretrained('{out}').to('cuda')")


if __name__ == "__main__":
    main()
