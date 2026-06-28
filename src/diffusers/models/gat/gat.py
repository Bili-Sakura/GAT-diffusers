from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional, Type

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