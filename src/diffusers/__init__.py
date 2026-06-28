from importlib.metadata import version as _package_version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    __version__ = _package_version("diffusers")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "GATDiscriminator",
    "GATGenerator",
    "GATPipeline",
    "GATD_models",
    "GAT_MODEL_PRESETS",
    "GAT_models",
    "RpGANLoss",
    "RpGANPTLoss",
    "apply_checkpoint_args",
    "convert_gat_checkpoint",
    "get_checkpoint_state",
    "get_gat_config",
    "load_encoders",
    "load_gat_generator_from_checkpoint",
    "load_gat_pipeline",
    "load_legacy_checkpoints",
    "normalize_model_name",
    "save_gat_pipeline_pretrained",
]


def __getattr__(name: str):
    if name == "GATPipeline":
        from .pipelines.gat.gat import GATPipeline

        return GATPipeline
    if name in {"GATGenerator", "GAT_models", "GATDiscriminator", "GATD_models"}:
        from .models.gat.gat import (
            GATDiscriminator,
            GATD_models,
            GATGenerator,
            GAT_models,
        )

        mapping = {
            "GATGenerator": GATGenerator,
            "GAT_models": GAT_models,
            "GATDiscriminator": GATDiscriminator,
            "GATD_models": GATD_models,
        }
        return mapping[name]
    if name in {"RpGANLoss", "RpGANPTLoss"}:
        from .gat_utils import gat as gat_utils

        return gat_utils.RpGANLoss if name == "RpGANLoss" else gat_utils.RpGANPTLoss
    if name in {
        "GAT_MODEL_PRESETS",
        "apply_checkpoint_args",
        "convert_gat_checkpoint",
        "get_checkpoint_state",
        "get_gat_config",
        "load_encoders",
        "load_gat_generator_from_checkpoint",
        "load_gat_pipeline",
        "load_legacy_checkpoints",
        "normalize_model_name",
        "save_gat_pipeline_pretrained",
    }:
        from .gat_utils import gat as gat_utils

        return getattr(gat_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
