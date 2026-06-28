from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .._hf import get_hf_attr, get_hf_diffusers
from ..models.gat.discriminator import GATDiscriminator
from ..models.gat.generator import GATGenerator, GAT_models
from .config import GAT_MODEL_PRESETS, get_gat_config, normalize_model_name
from .encoders import load_legacy_checkpoints


def get_checkpoint_state(checkpoint, weight_key: str = "ema"):
    for key in (weight_key, "ema", "generator", "model"):
        state_dict = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state_dict, dict):
            return state_dict, key
    raise RuntimeError("Checkpoint does not contain model weights.")


def apply_checkpoint_args(args, checkpoint):
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if ckpt_args is None:
        return args
    for name in ("model", "resolution", "num_classes", "fused_attn", "qk_norm"):
        if hasattr(ckpt_args, name):
            setattr(args, name, getattr(ckpt_args, name))
    args.model = normalize_model_name(args.model)
    return args


def load_gat_generator_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: Optional[str] = None,
    resolution: int = 256,
    num_classes: int = 1000,
    z_dims: Optional[list[int]] = None,
    weight_key: str = "ema",
    legacy: bool = False,
    encoder_depth: int = 8,
    fused_attn: bool = True,
    qk_norm: bool = True,
    device: str | torch.device = "cpu",
) -> GATGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_name is None and isinstance(checkpoint, dict) and checkpoint.get("args") is not None:
        model_name = normalize_model_name(checkpoint["args"].model)
    if model_name is None:
        raise ValueError("model_name is required when checkpoint does not store training args.")

    state_dict, _ = get_checkpoint_state(checkpoint, weight_key)
    if legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=encoder_depth)

    config = get_gat_config(model_name, resolution, num_classes=num_classes, z_dims=z_dims)
    generator = GAT_models[model_name](**config, fused_attn=fused_attn, qk_norm=qk_norm)
    generator.load_state_dict(state_dict, strict=True)
    return generator.to(device)


def load_gat_pipeline(
    checkpoint_path: str,
    *,
    model_name: Optional[str] = None,
    resolution: int = 256,
    vae: str = "ema",
    weight_key: str = "ema",
    truncation_psi: float = 0.3,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
):
    GATPipeline = get_hf_attr("diffusers.pipelines.gat.pipeline_gat.GATPipeline")
    AutoencoderKL = get_hf_attr("diffusers.models.autoencoder_kl.AutoencoderKL")

    generator = load_gat_generator_from_checkpoint(
        checkpoint_path,
        model_name=model_name,
        resolution=resolution,
        weight_key=weight_key,
        device=device,
    ).to(dtype=torch_dtype)

    vae_model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}").to(device=device, dtype=torch_dtype)
    get_hf_diffusers()
    return GATPipeline(generator=generator, vae=vae_model, truncation_psi=truncation_psi).to(device)
