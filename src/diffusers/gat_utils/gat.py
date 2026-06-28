from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .._hf import get_hf_attr, get_hf_diffusers
from ..models.gat.gat import GATDiscriminator, GATD_models, GATGenerator, GAT_models


GAT_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "GAT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "GAT-XL/4": {"depth": 28, "hidden_size": 1152, "patch_size": 4, "num_heads": 16},
    "GAT-XL/8": {"depth": 28, "hidden_size": 1152, "patch_size": 8, "num_heads": 16},
    "GAT-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "GAT-L/4": {"depth": 24, "hidden_size": 1024, "patch_size": 4, "num_heads": 16},
    "GAT-L/8": {"depth": 24, "hidden_size": 1024, "patch_size": 8, "num_heads": 16},
    "GAT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    "GAT-B/4": {"depth": 12, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    "GAT-B/8": {"depth": 12, "hidden_size": 768, "patch_size": 8, "num_heads": 12},
    "GAT-S/2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "GAT-S/4": {"depth": 12, "hidden_size": 384, "patch_size": 4, "num_heads": 6},
    "GAT-S/8": {"depth": 12, "hidden_size": 384, "patch_size": 8, "num_heads": 6},
}


def normalize_model_name(name: str) -> str:
    return name.replace("SiT-", "GAT-", 1) if name.startswith("SiT-") else name


def get_gat_config(model_name: str, resolution: int, num_classes: int = 1000, z_dims: list[int] | None = None) -> Dict[str, Any]:
    if model_name not in GAT_MODEL_PRESETS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {sorted(GAT_MODEL_PRESETS)}")
    preset = GAT_MODEL_PRESETS[model_name]
    latent_size = resolution // 8
    return {
        "input_size": latent_size,
        "num_classes": num_classes,
        "z_dims": z_dims or [768],
        **preset,
    }


@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    if resolution == 128:
        resolution = 256
    if resolution not in (256, 512):
        raise ValueError(f"Unsupported resolution: {resolution}")

    encoders, encoder_types, architectures = [], [], []
    for enc_name in enc_type.split(","):
        encoder_type, architecture, model_config = enc_name.split("-")
        if "dinov2" not in encoder_type:
            raise NotImplementedError("This refactor keeps only DINOv2 encoders.")

        import timm

        model_name = f"dinov2_vit{model_config}14_reg" if "reg" in encoder_type else f"dinov2_vit{model_config}14"
        encoder = torch.hub.load("facebookresearch/dinov2", model_name)
        del encoder.head
        patch_resolution = 16 * (resolution // 256)
        encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            encoder.pos_embed.data,
            [patch_resolution, patch_resolution],
        )
        encoder.head = torch.nn.Identity()
        encoder = encoder.to(device)
        encoder.eval()

        encoders.append(encoder)
        encoder_types.append(encoder_type)
        architectures.append(architecture)

    return encoders, encoder_types, architectures


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "decoder_blocks" in key:
            parts = key.split(".")
            parts[0] = "blocks"
            parts[1] = str(int(parts[1]) + encoder_depth)
            new_state_dict[".".join(parts)] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


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


def extract_generator_state_dict(checkpoint_path: str, weight_key: str = "ema", legacy: bool = False, encoder_depth: int = 8):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, used_key = get_checkpoint_state(checkpoint, weight_key)
    if legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=encoder_depth)
    return checkpoint, state_dict, used_key


def save_gat_pipeline_pretrained(
    output_dir: str | Path,
    generator: GATGenerator,
    *,
    truncation_psi: float = 0.3,
    vae_hub_id: str = "stabilityai/sd-vae-ft-ema",
    id2label: Optional[Dict[int, str]] = None,
    discriminator: Optional[GATDiscriminator] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.save_pretrained(output_dir / "generator")
    if discriminator is not None:
        discriminator.save_pretrained(output_dir / "discriminator")

    model_index = {
        "_class_name": "GATPipeline",
        "_diffusers_version": get_hf_diffusers().__version__,
        "generator": ["diffusers", "GATGenerator"],
        "vae": ["diffusers", "AutoencoderKL"],
        "truncation_psi": truncation_psi,
        "vae_hub_id": vae_hub_id,
    }
    if id2label:
        model_index["id2label"] = {str(k): v for k, v in id2label.items()}
    (output_dir / "model_index.json").write_text(json.dumps(model_index, indent=2), encoding="utf-8")
    return output_dir


def convert_gat_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    *,
    model_name: Optional[str] = None,
    resolution: int = 256,
    num_classes: int = 1000,
    z_dims: Optional[list[int]] = None,
    weight_key: str = "ema",
    legacy: bool = False,
    encoder_depth: int = 8,
    save_discriminator: bool = False,
    discriminator_weight_key: str = "discriminator",
    truncation_psi: float = 0.3,
    vae_hub_id: str = "stabilityai/sd-vae-ft-ema",
    fused_attn: bool = True,
    qk_norm: bool = True,
) -> Path:
    checkpoint, state_dict, _ = extract_generator_state_dict(
        checkpoint_path, weight_key=weight_key, legacy=legacy, encoder_depth=encoder_depth
    )
    if model_name is None and isinstance(checkpoint, dict) and checkpoint.get("args") is not None:
        model_name = normalize_model_name(checkpoint["args"].model)
    if model_name is None:
        raise ValueError("model_name is required when checkpoint does not store training args.")

    config = get_gat_config(model_name, resolution, num_classes=num_classes, z_dims=z_dims)
    generator = GAT_models[model_name](**config, fused_attn=fused_attn, qk_norm=qk_norm)
    generator.load_state_dict(state_dict, strict=True)

    discriminator = None
    if save_discriminator and isinstance(checkpoint, dict) and discriminator_weight_key in checkpoint:
        disc_state = checkpoint[discriminator_weight_key]
        discriminator = GATD_models[model_name](**config, fused_attn=fused_attn, qk_norm=qk_norm)
        discriminator.load_state_dict(disc_state, strict=True)

    return save_gat_pipeline_pretrained(
        output_dir,
        generator,
        truncation_psi=truncation_psi,
        vae_hub_id=vae_hub_id,
        discriminator=discriminator,
    )


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
    checkpoint, state_dict, _ = extract_generator_state_dict(
        checkpoint_path, weight_key=weight_key, legacy=legacy, encoder_depth=encoder_depth
    )
    if model_name is None and isinstance(checkpoint, dict) and checkpoint.get("args") is not None:
        model_name = normalize_model_name(checkpoint["args"].model)
    if model_name is None:
        raise ValueError("model_name is required when checkpoint does not store training args.")

    config = get_gat_config(model_name, resolution, num_classes=num_classes, z_dims=z_dims)
    generator = GAT_models[model_name](**config, fused_attn=fused_attn, qk_norm=qk_norm)
    generator.load_state_dict(state_dict, strict=True)
    return generator.to(device)


def load_gat_pipeline(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    **kwargs,
):
    GATPipeline = get_hf_attr("diffusers.pipelines.gat.gat.GATPipeline")
    pipe = GATPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    return pipe.to(device)


# --- DiffAugment ---
# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
import numpy as np

def DiffAugment(x, prob=1.0, policy='', channels_first=True, aug_params=None):
    if np.random.rand() > prob:
        return x, {}
    
    aug_params_new = {}
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)

        for p in policy.split(','):
            
            aug_params_new[p] = []
            
            if aug_params is None:
                for f in AUGMENT_FNS[p]:
                    x, _param = f(x)
                    aug_params_new[p].append(_param)
            else:
                for f, _param in zip(AUGMENT_FNS[p], aug_params[p]):
                    x, _param = f(x, _param)
                    aug_params_new[p].append(_param)
            
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x, aug_params_new


def rand_brightness(x, params=None):
    if params is None:
        noise = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (noise - 0.5)
        params = noise
    else:
        noise = params
        x = x + (noise - 0.5)
    return x, params


def rand_saturation(x, params=None):
    if params is None:
        x_mean, noise = x.mean(dim=1, keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (noise * 2) + x_mean
        params = (x_mean, noise)
    else:
        x_mean, noise = params
        x = (x - x_mean) * (noise * 2) + x_mean
    return x, params


def rand_contrast(x, params=None):
    if params is None:
        x_mean, noise = x.mean(dim=[1, 2, 3], keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (noise + 0.5) + x_mean
        params = (x_mean, noise)
    else:
        x_mean, noise = params
        x = (x - x_mean) * (noise + 0.5) + x_mean
    return x, params


def rand_brightness_saturation_contrast(x, params=None, p=0.5):
    
    if params is None:
        noise_bright = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x_mean_sat, noise_sat = x.mean(dim=1, keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        x_mean_cont, noise_cont = x.mean(dim=[1, 2, 3], keepdim=True), torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    
        mask = torch.rand(x.size(0), device=x.device)
        mask = mask < p
    else:
        mask, noise_bright, x_mean_sat, noise_sat, x_mean_cont, noise_cont = params
        
    x_aug = x + (noise_bright - 0.5)
    x_aug = (x_aug - x_mean_sat) * (noise_sat * 2) + x_mean_sat
    x_aug = (x_aug - x_mean_cont) * (noise_cont + 0.5) + x_mean_cont

    x_out = x.clone()
    x_out[mask] = x_aug[mask]
    
    params = [mask, noise_bright, x_mean_sat, noise_sat, x_mean_cont, noise_cont]

    return x_out, params
    


def rand_translation(x, params=None, ratio=0.125):
    if params is None:
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        params = (translation_x, translation_y, x.shape[2], x.shape[3])
    else:
        translation_x, translation_y, h, w = params
        translation_x, translation_y = translation_x * h // x.shape[2], translation_y * w // x.shape[3]
        
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x, params


def rand_cutout(x, params=None, ratio=0.5):
    if params is None:
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        params = (cutout_size, offset_x, offset_y, x.shape[2], x.shape[3])
    else:
        cutout_size, offset_x, offset_y, h, w = params
        cutout_size, offset_x, offset_y = (cutout_size[0] * h // x.shape[2], cutout_size[1] * w // x.shape[3]), offset_x * h // x.shape[2], offset_y * w // x.shape[3]
        
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, params


def rand_flip(x: torch.Tensor,
                    params: torch.Tensor | None = None,
                    ratio: float = 0.4,
                    dim: int = 3):
    B = x.size(0)

    if params is None:
        params = torch.rand(B, device=x.device)

    mask = params < ratio
    if mask.any():
        x_out = x.clone()
        x_out[mask] = x[mask].flip(dim)
        return x_out, params
    else:
        return x, params

AUGMENT_FNS = {
    'color': [rand_brightness_saturation_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
}


aug = DiffAugment

import torch
import numpy as np

# diffaug below

import math

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def info_per_layer(N, Imin=0.125):
    idx = np.arange(N)
    lam = idx / (N - 1)           
    return 1. - Imin ** (1 - lam)

class RpGANLoss:
    def __init__(
            self,
            encoders=[], 
            accelerator=None,
            r1_gamma=0.1,
            r1_every=8,
            r2_gamma=0.1,
            r2_every=8,
            approximate=False,
            ):
        self.encoders = encoders
        self.accelerator = accelerator
        
        
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.r1_every = r1_every
        self.r2_every = r2_every
        
        self.approximate = approximate
        self.policy = 'color,translation,cutout,flip'
        self.policy_raw_image = 'translation,flip'
        
        self.approximated_GP_std = 0.01
        self.aug_prob = 1.0
        
    def apply_gaussian(self, x, t, seed_noise=None):
        if seed_noise is None:
            seed_noise = torch.randn_like(x)
            
        return x * (1. - t) + seed_noise * t, seed_noise
    
    def apply_gaussian_list(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        noises = []
        
        if seed_noise is None:
            seed_noise = [None for _ in range(n_ts)]
        seed_noise = seed_noise[-n_ts:]
        noise_schedule = noise_schedule[-n_ts:]
        
        for i in range(n_ts):
            x_noised, seed_noise_ = self.apply_gaussian(xs[i], noise_schedule[i], seed_noise[i])
            xs_new.append(x_noised) 
            noises.append(seed_noise_)
            
        return torch.stack(xs_new, dim=0), torch.stack(noises, dim=0)
    
    def apply_gaussian_list_cumulative(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        
        if seed_noise is None:
            seed_noise = [torch.randn_like(xs[_]) for _ in range(n_ts)]
        
        cumulative_noise = torch.zeros_like(xs[0])
        
        prev_t_squared, prev_t = 0.0, 0.0

        noise_schedule = list(noise_schedule)[::-1]
        xs = list(xs)[::-1]
        
        for idx, t in enumerate(noise_schedule):

            current_t_squared = t**2
            
            decay_ratio = (1. - t) / (1. - prev_t)
            
            incremental_variance = current_t_squared - prev_t_squared * decay_ratio ** 2

            incremental_noise_sample = seed_noise[idx]
            
            scaled_incremental_noise = incremental_noise_sample * math.sqrt(incremental_variance)
            
            cumulative_noise = cumulative_noise * decay_ratio + scaled_incremental_noise
            
            x_noisy = (1 - t) * xs[idx] + cumulative_noise
            
            xs_new.append(x_noisy)
            
            prev_t_squared = current_t_squared
            prev_t = t

        xs_new = list(xs_new)[::-1]
        return torch.stack(xs_new, dim=0), seed_noise
    
    def approximated_gradient_penalty(self, data, pred, model, model_kwargs={}):

        aug_params, noise_params = model_kwargs["aug_params"], model_kwargs["noise_params"]
        
        if len(data.shape) > 4:
            
            data_list = []
            for idx, d in enumerate(data):
                d, _ = aug(d + torch.randn_like(d) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
                data_list.append(d)
            data = torch.stack(data_list, dim=0)
        else:
            data, _ = aug(data + torch.randn_like(data) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
            data = data.unsqueeze(0).repeat(len(noise_params), 1, 1, 1, 1)  
            
        data, _ = self.apply_gaussian_list_cumulative(data, noise_params)
        
        pred_noised = model(data, y=model_kwargs["y"])
        
        return ((pred - pred_noised) / self.approximated_GP_std).pow(2).mean(-1)
    
    
    def gradient_penalty(self, data, pred, model_kwargs={}):
        gradients = torch.autograd.grad(
            outputs=pred.sum(), inputs=data,
            create_graph=True, retain_graph=True)[0]
        gradient_penalty = gradients.pow(2).sum([1, 2, 3]).mean()
        return gradient_penalty
    
    def step_gen(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.latent_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
        
        
        gen_images  = generator(update_ema=True, multiscale=True, **model_kwargs)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
            
        
        gen_logits = discriminator(gen_images_aug, y=model_kwargs["y"])
        
        with torch.no_grad():
            real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        relativistic_logits = gen_logits - real_logits
        gen_loss = torch.nn.functional.softplus(-relativistic_logits).mean(-1)
        
        loss = gen_loss
        
        loss_dict = {
            "gen_loss": gen_loss,
        }
        extras = {
            "gen_images": gen_images
        }
        return loss, loss_dict, extras
    

    def step_disc(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        r1_gamma, r2_gamma = self.r1_gamma, self.r2_gamma
        aug_prob = 1.0
        
        
        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.latent_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
            
                                            
        with torch.no_grad():
            gen_images  = generator(update_ema=False, multiscale=True, **model_kwargs)
        
        
        if global_step % self.r1_every == 0:
            images = images.detach()
            images.requires_grad = True
        if global_step % self.r2_every == 0:
            gen_images = gen_images.detach()
            gen_images.requires_grad = True
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
        
        gen_logits = discriminator(gen_images_aug, y=model_kwargs["y"])
        real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        relativistic_logits_cls = real_logits - gen_logits
        disc_loss = torch.nn.functional.softplus(-relativistic_logits_cls).mean(-1)
        
        loss = disc_loss
        loss_dict = {
            "disc_loss": disc_loss,
        }
        
        assert len(real_logits.shape) == 2, "logit should be [B, nlogits] for current multi-scale pred implementation"
        if global_step % self.r1_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": real_aug_params, "noise_params": real_noise}
                r1_loss = self.approximated_gradient_penalty(images, real_logits, discriminator, model_kwargs_GP)
            else:
                r1_loss = self.gradient_penalty(images, real_logits.mean(-1), discriminator, model_kwargs)
            loss += r1_gamma / 2 * r1_loss
            loss_dict["r1_loss"] = r1_loss
        if global_step % self.r2_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": gen_aug_params, "noise_params": gen_noise}
                r2_loss = self.approximated_gradient_penalty(gen_images, gen_logits, discriminator, model_kwargs_GP)
            else:
                r2_loss = self.gradient_penalty(gen_images, gen_logits.mean(-1), discriminator, model_kwargs)
            loss += r2_gamma / 2 * r2_loss
            loss_dict["r2_loss"] = r2_loss
            
        extras = {}
        return loss, loss_dict, extras

import torch
import numpy as np

# diffaug below
from torchvision.transforms import Normalize

import math

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    return torch.sum(x, dim=list(range(1, len(x.size()))))


def info_per_layer(N, Imin=0.125):
    idx = np.arange(N)
    lam = idx / (N - 1)           
    return 1. - Imin ** (1 - lam)

class RpGANPTLoss:
    def __init__(
            self,
            encoders=[], 
            encoder_types=[],
            architectures=[],
            accelerator=None,
            r1_gamma=0.1,
            r1_every=8,
            r2_gamma=0.1,
            r2_every=8,
            proj_coeff=1.0,
            approximate=False,
            ):
        self.encoders = encoders
        self.encoder_types = encoder_types
        self.architectures = architectures
        self.accelerator = accelerator
        
        self.proj_coeff = proj_coeff
        
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma
        self.r1_every = r1_every
        self.r2_every = r2_every
        
        self.approximate = approximate
        self.policy = 'color,translation,cutout,flip'
        self.policy_raw_image = 'translation,flip'
        
        self.approximated_GP_std = 0.01
        self.aug_prob = 1.0
        
    def apply_gaussian(self, x, t, seed_noise=None):
        if seed_noise is None:
            seed_noise = torch.randn_like(x)
            
        return x * (1. - t) + seed_noise * t, seed_noise
    
    def apply_gaussian_list(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        noises = []
        
        if seed_noise is None:
            seed_noise = [None for _ in range(n_ts)]
        seed_noise = seed_noise[-n_ts:]
        noise_schedule = noise_schedule[-n_ts:]
        
        for i in range(n_ts):
            x_noised, seed_noise_ = self.apply_gaussian(xs[i], noise_schedule[i], seed_noise[i])
            xs_new.append(x_noised) 
            noises.append(seed_noise_)
            
        return torch.stack(xs_new, dim=0), torch.stack(noises, dim=0)
    
    def apply_gaussian_list_cumulative(self, xs, seed_noise=None, min_info=0.125):
        noise_schedule = info_per_layer(N=4, Imin=min_info)
        
        noise_schedule = noise_schedule + 1e-2
        
        n_ts = len(xs)
        xs_new = []
        
        if seed_noise is None:
            seed_noise = [torch.randn_like(xs[_]) for _ in range(n_ts)]
        
        cumulative_noise = torch.zeros_like(xs[0])
        
        prev_t_squared, prev_t = 0.0, 0.0

        noise_schedule = list(noise_schedule)[::-1]
        xs = list(xs)[::-1]
        
        for idx, t in enumerate(noise_schedule):

            current_t_squared = t**2
            
            decay_ratio = (1. - t) / (1. - prev_t)
            
            incremental_variance = current_t_squared - prev_t_squared * decay_ratio ** 2

            incremental_noise_sample = seed_noise[idx]
            
            scaled_incremental_noise = incremental_noise_sample * math.sqrt(incremental_variance)
            
            cumulative_noise = cumulative_noise * decay_ratio + scaled_incremental_noise
            
            x_noisy = (1 - t) * xs[idx] + cumulative_noise
            
            xs_new.append(x_noisy)
            
            prev_t_squared = current_t_squared
            prev_t = t

        xs_new = list(xs_new)[::-1]
        return torch.stack(xs_new, dim=0), seed_noise
    
    def approximated_gradient_penalty(self, data, pred, model, model_kwargs={}):

        aug_params, noise_params = model_kwargs["aug_params"], model_kwargs["noise_params"]
        
        if len(data.shape) > 4:
            
            data_list = []
            for idx, d in enumerate(data):
                d, _ = aug(d + torch.randn_like(d) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
                data_list.append(d)
            data = torch.stack(data_list, dim=0)
        else:
            data, _ = aug(data + torch.randn_like(data) * self.approximated_GP_std, aug_params=aug_params, policy=self.policy)
            data = data.unsqueeze(0).repeat(len(noise_params), 1, 1, 1, 1)  
            
        data, _ = self.apply_gaussian_list_cumulative(data, noise_params)
        
        pred_noised = model(data, y=model_kwargs["y"])
        
        return ((pred - pred_noised) / self.approximated_GP_std).pow(2).mean(-1)
    
    
    def gradient_penalty(self, data, pred, model_kwargs={}):
        gradients = torch.autograd.grad(
            outputs=pred.sum(), inputs=data,
            create_graph=True, retain_graph=True)[0]
        gradient_penalty = gradients.pow(2).sum([1, 2, 3]).mean()
        return gradient_penalty
    
    
    def encode_feature(self, raw_image, do_aug=True, aug_params=None):
        zs = []
        with self.accelerator.autocast():
            with torch.no_grad():
                for encoder, encoder_type, arch in zip(self.encoders, self.encoder_types, self.architectures):
                    raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                    
                    if do_aug:
                        raw_image_, _ = aug(raw_image_, aug_params=aug_params, policy=self.policy_raw_image)
                    
                    z = encoder.forward_features(raw_image_)
                    
                    assert 'dinov2' in encoder_type
                    
                    zs.append(z)
        return zs
    
    def step_gen(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.hidden_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
        
        
        gen_images  = generator(update_ema=True, multiscale=True, **model_kwargs)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images.detach(), prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
            
        
        gen_logits, aux = discriminator(gen_images_aug, y=model_kwargs["y"], return_aux=True)
        
        zs = aux["x_feat"]
        with torch.no_grad():
            real_logits = discriminator(images_aug, y=model_kwargs["y"])
        
        
        relativistic_logits = gen_logits - real_logits
        gen_loss = torch.nn.functional.softplus(-relativistic_logits).mean(-1)
        
        z_loss = zs[0].mean() * 0.0 + zs[1].mean() * 0.0
        loss = gen_loss + z_loss
        
        loss_dict = {
            "gen_loss": gen_loss,
        }
        extras = {
            "gen_images": gen_images
        }
        return loss, loss_dict, extras
    

    def step_disc(self, generator, discriminator, discriminator_ema, images, raw_images, global_step, model_kwargs=None, zs=None, **kwargs):

        r1_gamma, r2_gamma = self.r1_gamma, self.r2_gamma
        aug_prob = 1.0
        
        
        if model_kwargs == None:
            model_kwargs = {}
        if "z" not in model_kwargs.keys():
            z = torch.randn(images.shape[0], generator.module.latent_size, device=images.device, dtype=images.dtype)
            model_kwargs["z"] = z
        if "x" not in model_kwargs.keys():
            model_kwargs["x"] = torch.randn_like(images)
            
                                            
        with torch.no_grad():
            gen_images  = generator(update_ema=False, multiscale=True, **model_kwargs)
        
        
        if global_step % self.r1_every == 0:
            images = images.detach()
            images.requires_grad = True
        if global_step % self.r2_every == 0:
            gen_images = gen_images.detach()
            gen_images.requires_grad = True
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_aug_params = aug(gen_images[-1], prob=self.aug_prob, policy=self.policy)
            
            gen_images_aug_list = []
            for gen_img in gen_images[:-1]:
                gen_images_aug_list.append(aug(gen_img, aug_params=gen_aug_params, policy=self.policy)[0])
            gen_images_aug = torch.stack(gen_images_aug_list + [gen_images_aug], dim=0)
            
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        else:    
            gen_images_aug, gen_aug_params = aug(gen_images, prob=self.aug_prob, policy=self.policy)
            images_aug, real_aug_params = aug(images, prob=self.aug_prob, policy=self.policy)
        
        images_aug = images_aug.unsqueeze(0).repeat(len(gen_images_aug), 1, 1, 1, 1)
        
        if len(gen_images.shape) > 4:
            gen_images_aug, gen_noise = self.apply_gaussian_list_cumulative(gen_images_aug)
            images_aug, real_noise = self.apply_gaussian_list_cumulative(images_aug)
        
        gen_logits, gen_aux = discriminator(gen_images_aug, y=model_kwargs["y"], return_aux=True)
        real_logits, real_aux = discriminator(images_aug, y=model_kwargs["y"], return_aux=True)
        
        z_feat_tilde = real_aux["x_feat"]
        
        with torch.no_grad():
            z_feat = self.encode_feature(raw_images, do_aug=True, aug_params=real_aug_params)[0]
        
        z_cls_kd_loss = 1. - torch.nn.functional.cosine_similarity(z_feat_tilde[0].squeeze(1), z_feat['x_norm_clstoken'].detach(), dim=-1)
        
        
        z_feat_spatial = z_feat['x_norm_patchtokens']
        
        
        if z_feat_tilde[1].shape != z_feat_spatial.shape:
            z_feat_spatial = resize_spatial(z_feat_spatial, H_out=int(np.sqrt(z_feat_tilde[1].shape[1])))
        
        
        z_spatial_kd_loss = (1. - torch.nn.functional.cosine_similarity(z_feat_tilde[1], z_feat_spatial.detach(), dim=-1)).sum(dim=1) / z_feat_spatial.shape[1]
        
        z_kd_loss = z_cls_kd_loss + z_spatial_kd_loss
        
        relativistic_logits_cls = real_logits - gen_logits
        disc_loss = torch.nn.functional.softplus(-relativistic_logits_cls).mean(-1)
        
        loss = disc_loss + z_kd_loss * self.proj_coeff
        loss_dict = {
            "disc_loss": disc_loss,
            "z_kd_cls": z_cls_kd_loss,
            "z_kd_patch": z_spatial_kd_loss,
        }
        
        assert len(real_logits.shape) == 2, "logit should be [B, nlogits] for current multi-scale pred implementation"
        if global_step % self.r1_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": real_aug_params, "noise_params": real_noise}
                r1_loss = self.approximated_gradient_penalty(images, real_logits, discriminator, model_kwargs_GP)
            else:
                r1_loss = self.gradient_penalty(images, real_logits.mean(-1), discriminator, model_kwargs)
            loss += r1_gamma / 2 * r1_loss
            loss_dict["r1_loss"] = r1_loss
        if global_step % self.r2_every == 0:
            if self.approximate:
                model_kwargs_GP = {"y": model_kwargs["y"], "aug_params": gen_aug_params, "noise_params": gen_noise}
                r2_loss = self.approximated_gradient_penalty(gen_images, gen_logits, discriminator, model_kwargs_GP)
            else:
                r2_loss = self.gradient_penalty(gen_images, gen_logits.mean(-1), discriminator, model_kwargs)
            loss += r2_gamma / 2 * r2_loss
            loss_dict["r2_loss"] = r2_loss
            
        extras = {}
        return loss, loss_dict, extras
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x


def resize_spatial(tensor: torch.Tensor, H_out: int, W_out: int = None):
    B, L, C = tensor.shape
    H_in = W_in = int(np.sqrt(L))
    assert H_in * W_in == L, f"L={L} cannot be reshaped to square"

    if W_out is None:
        W_out = H_out
    
    x = tensor.transpose(1, 2).reshape(B, C, H_in, W_in)
    x = torch.nn.functional.interpolate(x, size=(H_out, W_out), mode='bicubic', align_corners=False)
    x = x.reshape(B, C, -1).transpose(1, 2)
    
    return x
