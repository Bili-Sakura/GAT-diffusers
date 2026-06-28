import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from diffusers import GATPipeline, GAT_models, normalize_model_name
from diffusers.models.gat.gat import apply_checkpoint_args, get_checkpoint_state, load_legacy_checkpoints
from diffusers._hf import get_hf_attr

AutoencoderKL = get_hf_attr("diffusers.models.autoencoder_kl.AutoencoderKL")


def create_npz_from_sample_folder(sample_dir, num):
    samples = []
    for i in tqdm(range(num), desc="Building npz"):
        samples.append(np.asarray(Image.open(f"{sample_dir}/{i:06d}.png")).astype(np.uint8))
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=np.stack(samples))
    print(f"Saved {npz_path}.")


def load_pipeline(args, device):
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir() and (ckpt_path / "model_index.json").exists():
        if dist.get_rank() == 0:
            print(f"Loading Diffusers pipeline from {ckpt_path}")
        return GATPipeline.from_pretrained(str(ckpt_path), torch_dtype=torch.float32).to(device)

    checkpoint = torch.load(args.ckpt, weights_only=False, map_location=device)
    args = apply_checkpoint_args(args, checkpoint)
    state_dict, state_key = get_checkpoint_state(checkpoint, args.weight_key)
    if args.legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=args.encoder_depth)

    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    generator = GAT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(",")],
        **block_kwargs,
    ).to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if dist.get_rank() == 0:
        print(f"Loaded legacy {state_key}: model={args.model}, resolution={args.resolution}")
    return GATPipeline(generator=generator, vae=vae, truncation_psi=args.truncation_psi).to(device)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires CUDA.")
    if args.ckpt is None:
        raise ValueError("--ckpt is required.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + rank)

    pipe = load_pipeline(args, device)
    model = pipe.generator
    vae = pipe.vae
    latent_size = int(model.config.input_size)

    latents_scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.zeros(1, 4, 1, 1, device=device)

    if rank == 0:
        print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_label = os.path.basename(str(args.ckpt).rstrip("/"))
    folder_name = (
        f"{getattr(args, 'model', 'GAT')}-{ckpt_label.replace('.pt', '')}"
        f"-size-{args.resolution}-vae-{args.vae}-seed-{args.global_seed}"
    )
    sample_folder = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder, exist_ok=True)
    dist.barrier()

    per_rank_batch = args.per_proc_batch_size
    global_batch = per_rank_batch * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch) * global_batch)
    iterations = total_samples // global_batch
    total = 0
    pbar = tqdm(range(iterations), disable=(rank != 0))

    for _ in pbar:
        x = torch.randn(per_rank_batch, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (per_rank_batch,), device=device)
        z = torch.randn(per_rank_batch, model.latent_size, device=device)
        output = model(x=x, y=y, z=z, truncation_psi=args.truncation_psi, return_dict=True).sample
        images = vae.decode((output - latents_bias) / latents_scale).sample
        images = (images + 1) / 2
        images = torch.clamp(255 * images, 0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

        for i, image in enumerate(images):
            index = total + i * world_size + rank
            if index < args.num_fid_samples:
                Image.fromarray(image).save(f"{sample_folder}/{index:06d}.png")
        total += global_batch

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder, args.num_fid_samples)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Legacy .pt file or converted Diffusers pipeline folder.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--weight-key", type=str, choices=["ema", "generator", "model"], default="ema")
    parser.add_argument("--model", type=str, choices=list(GAT_models.keys()), default="GAT-S/4")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--truncation-psi", type=float, default=0.3)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    main(parser.parse_args())
