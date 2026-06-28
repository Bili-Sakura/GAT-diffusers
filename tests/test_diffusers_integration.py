import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

torch = pytest.importorskip("torch")

from diffusers.models.gat.discriminator import GATDiscriminator
from diffusers.models.gat.generator import GATGenerator
from diffusers.pipelines.gat.pipeline_gat import GATPipeline


def test_gat_generator_forward_shape():
    model = GATGenerator(
        input_size=8,
        patch_size=2,
        in_channels=4,
        latent_size=16,
        hidden_size=32,
        depth=4,
        num_heads=4,
        num_classes=10,
        z_dims=[16],
        fused_attn=False,
        qk_norm=False,
    )
    x = torch.randn(2, 4, 8, 8)
    y = torch.tensor([1, 2])
    z = torch.randn(2, 16)
    output = model(x=x, y=y, z=z, return_dict=True).sample
    assert output.shape == x.shape


def test_gat_discriminator_forward_shape():
    model = GATDiscriminator(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=4,
        num_heads=4,
        num_classes=10,
        z_dims=[16],
        fused_attn=False,
        qk_norm=False,
    )
    x = [torch.randn(2, 4, 8, 8) for _ in range(4)]
    y = torch.tensor([1, 2])
    output = model(x, y, return_dict=True).logits
    assert output.shape == (2, 1)


def test_gat_pipeline_instantiation():
    generator = GATGenerator(
        input_size=8,
        patch_size=2,
        in_channels=4,
        latent_size=16,
        hidden_size=32,
        depth=4,
        num_heads=4,
        num_classes=10,
        z_dims=[16],
        fused_attn=False,
        qk_norm=False,
    )

    class DummyVAE:
        class config:
            block_out_channels = [128, 256, 512, 512]
            scaling_factor = 0.18215

        def decode(self, latents):
            class Out:
                sample = torch.zeros(latents.shape[0], 3, latents.shape[2] * 8, latents.shape[3] * 8)

            return Out()

    pipe = GATPipeline(generator=generator, vae=DummyVAE())
    assert pipe.generator is generator
