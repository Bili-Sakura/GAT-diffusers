#!/usr/bin/env python3
"""Smoke-test a converted Diffusers GAT pipeline folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from diffusers import GATPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_dir", type=str, help="Converted Diffusers pipeline directory.")
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = GATPipeline.from_pretrained(args.pipeline_dir, torch_dtype=torch.float32)
    pipe = pipe.to(args.device)
    with torch.inference_mode():
        out = pipe(class_labels=args.class_label, output_type="pt")
    print("sample shape:", out.images.shape)


if __name__ == "__main__":
    main()
