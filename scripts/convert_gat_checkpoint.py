#!/usr/bin/env python3
"""Convert a legacy GAT training checkpoint (.pt) to a Diffusers-style pipeline folder."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusers import GATGenerator, GAT_MODEL_PRESETS

TEMPLATES = ROOT / "hub_templates"
DEFAULT_LABELS = REPO_ROOT / "src/labels/id2label_en.json"
DEFAULT_VAE_DIR = REPO_ROOT / "models/BiliSakura/DiCo-diffusers/DiCo-XL-256/vae"


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
    parser.add_argument("--skip-vae", action="store_true", help="Skip copying bundled VAE weights.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Path to id2label JSON.")
    parser.add_argument("--vae-dir", type=Path, default=DEFAULT_VAE_DIR, help="Bundled VAE source directory.")
    return parser.parse_args()


def load_id2label(labels_path: Path) -> dict[str, str]:
    if not labels_path.is_file():
        return {}
    return json.loads(labels_path.read_text(encoding="utf-8"))


def materialize_vae(output_dir: Path, vae_dir: Path) -> None:
    target = output_dir / "vae"
    if (target / "diffusion_pytorch_model.safetensors").exists():
        return
    if not vae_dir.is_dir():
        raise FileNotFoundError(
            f"Bundled VAE not found at {vae_dir}. Convert a DiCo variant first or pass --vae-dir."
        )
    shutil.copytree(vae_dir, target)


def make_self_contained_repo(
    output_dir: Path,
    *,
    id2label: dict[str, str],
    truncation_psi: float,
    vae_hub_id: str,
) -> None:
    generator_dir = output_dir / "generator"
    shutil.copy2(TEMPLATES / "modeling_gat.py", generator_dir / "modeling_gat.py")
    shutil.copy2(TEMPLATES / "pipeline_gat.py", output_dir / "pipeline.py")

    config_path = generator_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_class_name"] = "GATGenerator"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model_index = {
        "_class_name": ["pipeline", "GATPipeline"],
        "_diffusers_version": "0.38.0",
        "generator": ["modeling_gat", "GATGenerator"],
        "vae": ["diffusers", "AutoencoderKL"],
        "truncation_psi": truncation_psi,
    }
    if id2label:
        model_index["id2label"] = id2label
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    vae_hub_id = f"stabilityai/sd-vae-ft-{args.vae}"
    id2label = load_id2label(args.labels)

    out = GATGenerator.convert_checkpoint(
        args.ckpt,
        output_dir,
        model_name=args.model,
        resolution=args.resolution,
        weight_key=args.weight_key,
        legacy=args.legacy,
        encoder_depth=args.encoder_depth,
        save_discriminator=args.save_discriminator,
        truncation_psi=args.truncation_psi,
        vae_hub_id=vae_hub_id,
        id2label={int(k): v for k, v in id2label.items()} if id2label else None,
    )

    if not args.skip_vae:
        materialize_vae(out, args.vae_dir)
    make_self_contained_repo(
        out,
        id2label=id2label,
        truncation_psi=args.truncation_psi,
        vae_hub_id=vae_hub_id,
    )

    print(f"Saved Diffusers pipeline to {out.resolve()}")
    print("Load with:")
    print("  from diffusers import DiffusionPipeline")
    print("  pipe = DiffusionPipeline.from_pretrained(")
    print(f"      '{out}', custom_pipeline='{out / 'pipeline.py'}', trust_remote_code=True")
    print("  ).to('cuda')")


if __name__ == "__main__":
    main()
