from __future__ import annotations

import json
import math
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

from ..._hf import (
    get_base_output,
    get_config_mixin,
    get_hf_attr,
    get_hf_diffusers,
    get_model_mixin,
    get_register_to_config,
)

ConfigMixin = get_config_mixin()
register_to_config = get_register_to_config()
ModelMixin = get_model_mixin()
BaseOutput = get_base_output()

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mult=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.register_parameter("bias", None)
        self.lr_mult = lr_mult
        self.init_weight(lr_mult=lr_mult)

    def init_weight(self, lr_mult):
        nn.init.xavier_uniform_(self.weight, gain=1 / lr_mult)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        bias = self.bias * self.lr_mult if self.bias is not None else None
        return torch.nn.functional.linear(x, self.weight * self.lr_mult, bias=bias)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.RMSNorm,
        fused_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, return_attention=False):
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn and not return_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            attn = None
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v

        x = x.transpose(1, 2).reshape(bsz, num_tokens, channels)
        x = self.proj_drop(self.proj(x))
        if return_attention:
            return x, attn
        return x


class FourierFeature(nn.Module):
    def __init__(self, hidden_size, resolution=16):
        super().__init__()
        self.linear = nn.Linear(2, hidden_size)
        y = torch.linspace(-1, 1, steps=resolution)
        x = torch.linspace(-1, 1, steps=resolution)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).view(1, resolution * resolution, 2)
        self.register_buffer("coords", coords)

    def reset_parameters(self):
        nn.init.uniform_(self.linear.weight, -np.sqrt(9 / 2), np.sqrt(9 / 2))

    def forward(self, x):
        return torch.sin(self.linear(self.coords.to(x.dtype))).repeat(x.shape[0], 1, 1)


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        self.pt_seq_len = pt_seq_len

        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        t_spatial = t[:, :, -self.pt_seq_len**2 :]
        t_pe = t_spatial * self.freqs_cos + rotate_half(t_spatial) * self.freqs_sin
        return torch.cat([t[:, :, : -self.pt_seq_len**2], t_pe], dim=2)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulate(x, scale, shift=None):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class GATGeneratorOutput(BaseOutput):
    sample: torch.Tensor


class _GenLabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train):
        return self.embedding_table(labels)


class GATBlock(nn.Module):
    def __init__(self, hidden_size, w_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            fused_attn=block_kwargs["fused_attn"],
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        use_swiglu = True
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(w_dim, 4 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, w_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(w_dim, hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        return self.linear(x)


class GATGenerator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        latent_size: int = 64,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        num_classes: int = 1000,
        z_dims: Optional[list[int]] = None,
        projector_dim: int = 2048,
        fused_attn: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        z_dims = z_dims or [768]
        block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm}

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        w_dim = self.hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = _GenLabelEmbedder(num_classes, latent_size, class_dropout_prob)
        self.num_patches = (input_size // patch_size) ** 2

        st_resolution = self.input_size // self.patch_size
        self.num_st_patches = st_resolution**2

        self.use_fourierfeat = False
        if self.use_fourierfeat:
            self.pos_embed = FourierFeature(hidden_size, resolution=st_resolution)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, st_resolution**2, hidden_size), requires_grad=False)

        self.use_rope = True
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.latent_embedder = nn.Sequential(
            EqualLinear(latent_size * 2, w_dim, lr_mult=0.01),
            nn.SiLU(),
            EqualLinear(w_dim, w_dim, lr_mult=0.01),
        )

        rgb_every = self.depth // 4
        self.rgb_indice = [(i + 1) * rgb_every - 1 for i in range(depth // rgb_every)]

        self.blocks = nn.ModuleList(
            [GATBlock(hidden_size, w_dim, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)]
        )
        self.final_layers = nn.ModuleList(
            [FinalLayer(hidden_size, w_dim, patch_size, self.out_channels) for _ in range(len(self.rgb_indice))]
        )

        self.initialize_weights()

        self.w_avg_beta = 0.995
        self.register_buffer("w_avg", torch.zeros([num_classes, w_dim]))

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_fourierfeat:
            self.pos_embed.reset_parameters()
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_st_patches**0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        layer_gain = 1e-1
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight, gain=layer_gain)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for final_layer in self.final_layers:
            nn.init.xavier_uniform_(final_layer.adaLN_modulation[-1].weight, gain=layer_gain)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)

        if isinstance(self.x_embedder, PatchEmbed):
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

    def unpatchify(self, x, patch_size=None):
        c = x.shape[-1] // (self.patch_size**2)
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def forward(
        self,
        x,
        y,
        z,
        guidance_scale=1.0,
        update_ema=False,
        truncation_psi=0.0,
        multiscale=False,
        return_dict: bool = False,
    ):
        x = self.x_embedder(x) * 0.0 + self.pos_embed(z) if self.use_fourierfeat else self.pos_embed + self.x_embedder(x) * 0.0

        y_idx = y
        y = self.y_embedder(y, self.training)
        c = torch.cat([normalize_2nd_moment(y.squeeze(1)), normalize_2nd_moment(z)], dim=1)
        c = self.latent_embedder(c)

        if truncation_psi != 0.0:
            c = c.lerp(self.w_avg[y_idx], truncation_psi)

        if self.w_avg_beta is not None and update_ema:
            unique_labels = y_idx.unique()
            for label in unique_labels:
                mask = y_idx.squeeze() == label
                avg_c = c[mask].mean(dim=0)
                self.w_avg[label].copy_(avg_c.detach().lerp(self.w_avg[label].to(c.dtype), self.w_avg_beta))

        if torch.is_tensor(guidance_scale):
            scales = guidance_scale
        else:
            scales = torch.full((y.shape[0],), guidance_scale, device=y.device, dtype=z.dtype)

        if (scales != 1.0).any():
            indices = torch.arange(self.num_classes).to(y_idx.device)
            y_null = self.y_embedder(indices, self.training)
            y_null = y_null.unsqueeze(0).repeat(z.shape[0], 1, 1)
            z_null = z.unsqueeze(1).repeat(1, self.num_classes, 1)
            w_null = torch.cat([normalize_2nd_moment(y_null), normalize_2nd_moment(z_null)], dim=-1)
            w_null = self.latent_embedder(w_null).detach().mean(dim=1)
            c = w_null + scales.unsqueeze(1) * (c - w_null)

        xs = []
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block), x, c, self.feat_rope, use_reentrant=False
            )
            xs.append(x)

        self.recent_x_std = x.std()

        rgbs = []
        rgb_accum = 0
        for i, final_layer in zip(self.rgb_indice, self.final_layers):
            rgb = final_layer(xs[i], c)
            rgb_accum = rgb_accum + self.unpatchify(rgb)
            rgbs.append(rgb_accum)

        if multiscale:
            output = torch.stack(rgbs, dim=0)
        else:
            output = rgbs[-1]

        if not return_dict:
            return (output,)
        return GATGeneratorOutput(sample=output)

    @classmethod
    def from_legacy_checkpoint(
        cls,
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
    ) -> "GATGenerator":
        return load_gat_generator_from_checkpoint(
            checkpoint_path,
            model_name=model_name,
            resolution=resolution,
            num_classes=num_classes,
            z_dims=z_dims,
            weight_key=weight_key,
            legacy=legacy,
            encoder_depth=encoder_depth,
            fused_attn=fused_attn,
            qk_norm=qk_norm,
            device=device,
        )

    @classmethod
    def convert_checkpoint(cls, checkpoint_path: str, output_dir: str, **kwargs) -> Path:
        return convert_gat_checkpoint(checkpoint_path, output_dir, **kwargs)


def GAT_XL_2(**kwargs):
    return GATGenerator(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def GAT_XL_4(**kwargs):
    return GATGenerator(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def GAT_XL_8(**kwargs):
    return GATGenerator(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def GAT_L_2(**kwargs):
    return GATGenerator(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def GAT_L_4(**kwargs):
    return GATGenerator(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def GAT_L_8(**kwargs):
    return GATGenerator(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def GAT_B_2(**kwargs):
    return GATGenerator(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def GAT_B_4(**kwargs):
    return GATGenerator(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def GAT_B_8(**kwargs):
    return GATGenerator(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def GAT_S_2(**kwargs):
    return GATGenerator(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def GAT_S_4(**kwargs):
    return GATGenerator(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def GAT_S_8(**kwargs):
    return GATGenerator(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


GAT_models = {
    "GAT-XL/2": GAT_XL_2,
    "GAT-XL/4": GAT_XL_4,
    "GAT-XL/8": GAT_XL_8,
    "GAT-L/2": GAT_L_2,
    "GAT-L/4": GAT_L_4,
    "GAT-L/8": GAT_L_8,
    "GAT-B/2": GAT_B_2,
    "GAT-B/4": GAT_B_4,
    "GAT-B/8": GAT_B_8,
    "GAT-S/2": GAT_S_2,
    "GAT-S/4": GAT_S_4,
    "GAT-S/8": GAT_S_8,
}


@dataclass
class GATDiscriminatorOutput(BaseOutput):
    logits: torch.Tensor


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class _DiscLabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.latent_embedder = nn.Sequential(
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01),
            nn.SiLU(),
            EqualLinear(hidden_size, hidden_size, lr_mult=0.01),
        )

    def forward(self, labels, train):
        embeddings = self.embedding_table(labels)
        return self.latent_embedder(embeddings)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layerscale=1e-1, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=block_kwargs["qk_norm"],
            fused_attn=block_kwargs["fused_attn"],
        )
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        use_swiglu = True
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.ls_attn = nn.Parameter(torch.ones(hidden_size) * layerscale)
        self.ls_mlp = nn.Parameter(torch.ones(hidden_size) * layerscale)

    def forward(self, x, c=None, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope) * self.ls_attn
        x = x + self.mlp(self.norm2(x)) * self.ls_mlp
        return x


class GATDiscriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        decoder_hidden_size: int = 768,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        num_classes: int = 1000,
        use_cfg: bool = False,
        z_dims: Optional[list[int]] = None,
        projector_dim: int = 2048,
        cmap_dim: int = 2048,
        fused_attn: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        z_dims = z_dims or [768]
        block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm}

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels * 4, hidden_size, bias=True)
        self.y_embedder = _DiscLabelEmbedder(num_classes, cmap_dim, class_dropout_prob)
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        layer_gain = 1e-1
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, layerscale=layer_gain, **block_kwargs)
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(hidden_size, cmap_dim, bias=True),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.aux_feat_size = z_dims[0]
        if self.aux_feat_size > 0:
            self.proj = build_mlp(hidden_size, projector_dim, z_dims[0])

        self.use_rope = True
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.feat_rope = None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    def unpatchify(self, x, patch_size=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def forward_encoder(self, xs, y):
        x = torch.cat([item for item in xs], dim=1)
        x = self.x_embedder(x) + self.pos_embed
        cls_token = self.cls_token.repeat([x.shape[0], 1, 1])
        x = torch.cat([cls_token, x], dim=1)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                self.ckpt_wrapper(block), x, y, self.feat_rope, use_reentrant=False
            )
        return x

    def forward(self, x, y, t=None, guidance_scale=1.0, return_aux=False, return_dict: bool = False):
        y = self.y_embedder(y, self.training)
        y = y.squeeze(dim=1)
        x = self.forward_encoder(x, y)
        x_cls, x_spatial = x[:, :1], x[:, 1:]
        x_logit = (self.final_layer(x_cls) * y.unsqueeze(1)).sum(-1)

        self.recent_x_std = x.std()

        if self.aux_feat_size > 0:
            x_feat_spatial = self.proj(x_spatial)
            x_feat_cls = self.proj(x_cls)
            x_feat = [x_feat_cls, x_feat_spatial]
        else:
            x_feat = None

        if return_aux:
            aux = {"x_feat": x_feat}
            if not return_dict:
                return x_logit, aux
            return GATDiscriminatorOutput(logits=x_logit)

        if not return_dict:
            return x_logit
        return GATDiscriminatorOutput(logits=x_logit)


def GAT_XL_2(**kwargs):
    return GATDiscriminator(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def GAT_XL_4(**kwargs):
    return GATDiscriminator(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def GAT_XL_8(**kwargs):
    return GATDiscriminator(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def GAT_L_2(**kwargs):
    return GATDiscriminator(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def GAT_L_4(**kwargs):
    return GATDiscriminator(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def GAT_L_8(**kwargs):
    return GATDiscriminator(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def GAT_B_2(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def GAT_B_4(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def GAT_B_8(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def GAT_S_2(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def GAT_S_4(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def GAT_S_8(**kwargs):
    return GATDiscriminator(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


GATD_models = {
    "GAT-XL/2": GAT_XL_2,
    "GAT-XL/4": GAT_XL_4,
    "GAT-XL/8": GAT_XL_8,
    "GAT-L/2": GAT_L_2,
    "GAT-L/4": GAT_L_4,
    "GAT-L/8": GAT_L_8,
    "GAT-B/2": GAT_B_2,
    "GAT-B/4": GAT_B_4,
    "GAT-B/8": GAT_B_8,
    "GAT-S/2": GAT_S_2,
    "GAT-S/4": GAT_S_4,
    "GAT-S/8": GAT_S_8,
}

# --- config, checkpoint I/O, training helpers ---

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

    from ...pipelines.gat.gat import save_gat_pipeline_pretrained
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



# --- DiffAugment ---
# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

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
