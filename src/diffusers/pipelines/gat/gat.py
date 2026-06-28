from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..._hf import get_hf_attr
from ...models.gat.gat import GATGenerator



def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
    if not id2label:
        return {}
    return {int(key): value for key, value in id2label.items()}


def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
    if not variant_path:
        return {}
    model_index_path = Path(variant_path).resolve() / "model_index.json"
    if not model_index_path.exists():
        return {}
    raw = json.loads(model_index_path.read_text(encoding="utf-8"))
    id2label = raw.get("id2label")
    if not isinstance(id2label, dict):
        return {}
    return {int(key): value for key, value in id2label.items()}


def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
    label2id: Dict[str, int] = {}
    for class_id, value in id2label.items():
        for synonym in value.split(","):
            synonym = synonym.strip()
            if synonym:
                label2id[synonym] = int(class_id)
    return dict(sorted(label2id.items()))


def _normalize_class_labels(
    class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
    *,
    device: torch.device,
    label2id: Dict[str, int],
) -> torch.LongTensor:
    if torch.is_tensor(class_labels):
        return class_labels.to(device=device, dtype=torch.long).reshape(-1)
    if isinstance(class_labels, int):
        class_label_ids = [class_labels]
    elif isinstance(class_labels, str):
        if not label2id:
            raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
        if class_labels not in label2id:
            raise ValueError(f"Unknown English label: {class_labels}")
        class_label_ids = [label2id[class_labels]]
    elif class_labels and isinstance(class_labels[0], str):
        if not label2id:
            raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
        missing = [item for item in class_labels if item not in label2id]
        if missing:
            raise ValueError(f"Unknown English label(s): {missing}")
        class_label_ids = [label2id[item] for item in class_labels]
    else:
        class_label_ids = list(class_labels)
    return torch.tensor(class_label_ids, device=device, dtype=torch.long).reshape(-1)


def _get_label_ids(label: Union[str, List[str]], label2id: Dict[str, int]) -> List[int]:
    labels = [label] if isinstance(label, str) else label
    if not label2id:
        raise ValueError("No English labels loaded. Provide `id2label` in the pipeline config.")
    missing = [item for item in labels if item not in label2id]
    if missing:
        preview = ", ".join(list(label2id.keys())[:8])
        raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
    return [label2id[item] for item in labels]

DiffusionPipeline = get_hf_attr("diffusers.pipelines.pipeline_utils.DiffusionPipeline")
ImagePipelineOutput = get_hf_attr("diffusers.pipelines.pipeline_utils.ImagePipelineOutput")
VaeImageProcessor = get_hf_attr("diffusers.image_processor.VaeImageProcessor")
randn_tensor = get_hf_attr("diffusers.utils.torch_utils.randn_tensor")


class GATPipeline(DiffusionPipeline):
    r"""
    Pipeline for one-step class-conditional image generation with Generative Adversarial Transformers (GAT).

    GAT generates images in a compact VAE latent space with a pure transformer generator trained
    adversarially with Multi-level Noise-perturbed image Guidance (MNG).
    """

    model_cpu_offload_seq = "generator->vae"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        truncation_psi = kwargs.pop("truncation_psi", None)
        vae_id = kwargs.pop("vae", None)
        generator_subfolder = kwargs.pop("generator_subfolder", "generator")
        base_path = Path(pretrained_model_name_or_path)

        if (base_path / "model_index.json").exists():
            try:
                pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            except Exception:
                generator_path = base_path / generator_subfolder
                generator = GATGenerator.from_pretrained(str(generator_path), torch_dtype=torch_dtype, **kwargs)
                id2label = _read_id2label_from_model_index(str(base_path))
                index = json.loads((base_path / "model_index.json").read_text(encoding="utf-8"))
                vae_id = vae_id or index.get("vae_hub_id", "stabilityai/sd-vae-ft-ema")
                truncation_psi = truncation_psi if truncation_psi is not None else index.get("truncation_psi", 0.3)
                AutoencoderKL = get_hf_attr("diffusers.models.autoencoder_kl.AutoencoderKL")
                vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch_dtype)
                pipe = cls(generator=generator, vae=vae, truncation_psi=truncation_psi, id2label=id2label)
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not a Diffusers GAT pipeline folder "
                "(expected model_index.json). Use scripts/convert_gat_checkpoint.py to convert a legacy .pt file."
            )

        if torch_dtype is not None:
            pipe = pipe.to(dtype=torch_dtype)
        return pipe


    def __init__(
        self,
        generator: GATGenerator,
        vae,
        truncation_psi: float = 0.3,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(generator=generator, vae=vae)
        self.register_to_config(truncation_psi=truncation_psi)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._id2label = _normalize_id2label(id2label)
        self.labels = _build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = _read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = _build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    def _get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        return _get_label_ids(label, self.labels)

    def _default_image_size(self) -> int:
        return int(self.generator.config.input_size) * self.vae_scale_factor

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if latents is not None:
            noise = latents.to(device=device, dtype=dtype)
        else:
            latent_height = height // self.vae_scale_factor
            latent_width = width // self.vae_scale_factor
            in_channels = int(getattr(self.generator, "in_channels", 4))
            noise = randn_tensor(
                (batch_size, in_channels, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        latent_size = int(getattr(self.generator, "latent_size", 64))
        z = randn_tensor(
            (batch_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return noise, z, noise

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if output_type == "latent":
            return latents
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        image = self.vae.decode(latents / scaling_factor).sample
        if output_type == "pt":
            return image
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]], torch.LongTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        truncation_psi: Optional[float] = None,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        default_size = self._default_image_size()
        height = int(height or default_size)
        width = int(width or default_size)
        truncation_psi = self.config.truncation_psi if truncation_psi is None else truncation_psi

        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt', 'latent'.")

        device = getattr(self, "_execution_device", None) or next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        class_labels_tensor = _normalize_class_labels(
            class_labels,
            device=device,
            label2id=self.labels,
        )
        batch_size = class_labels_tensor.shape[0]

        x, z, latents = self.prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        output = self.generator(
            x=x,
            y=class_labels_tensor,
            z=z,
            guidance_scale=guidance_scale,
            truncation_psi=truncation_psi,
            return_dict=True,
        ).sample

        if output_type == "latent":
            result = output
        else:
            result = self.decode_latents(output, output_type=output_type)

        if not return_dict:
            return (result,)
        return ImagePipelineOutput(images=result)


def save_gat_pipeline_pretrained(
    output_dir: str | Path,
    generator: GATGenerator,
    *,
    truncation_psi: float = 0.3,
    vae_hub_id: str = "stabilityai/sd-vae-ft-ema",
    id2label: Optional[Dict[Union[int, str], str]] = None,
    discriminator=None,
):
    from ..._hf import get_hf_diffusers
    from ...models.gat.gat import GATDiscriminator

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
