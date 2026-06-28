from importlib.metadata import version as _package_version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import importlib

importlib.import_module(f"{__name__}.pipelines.gat.pipeline_gat")

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
    "apply_checkpoint_args",
    "get_checkpoint_state",
    "get_gat_config",
    "load_encoders",
    "load_gat_generator_from_checkpoint",
    "load_gat_pipeline",
    "load_legacy_checkpoints",
    "normalize_model_name",
    "RpGANLoss",
    "RpGANPTLoss",
]


def __getattr__(name: str):
    if name == "GATPipeline":
        from .pipelines.gat.pipeline_gat import GATPipeline

        return GATPipeline
    if name in {"GATGenerator", "GAT_models"}:
        from .models.gat.generator import GATGenerator, GAT_models

        return GATGenerator if name == "GATGenerator" else GAT_models
    if name in {"GATDiscriminator", "GATD_models"}:
        from .models.gat.discriminator import GATDiscriminator, GATD_models

        return GATDiscriminator if name == "GATDiscriminator" else GATD_models
    if name in {"RpGANLoss", "RpGANPTLoss"}:
        from .gat_utils.losses import RpGANLoss, RpGANPTLoss

        return RpGANLoss if name == "RpGANLoss" else RpGANPTLoss
    if name in {
        "GAT_MODEL_PRESETS",
        "apply_checkpoint_args",
        "get_checkpoint_state",
        "get_gat_config",
        "load_encoders",
        "load_gat_generator_from_checkpoint",
        "load_gat_pipeline",
        "load_legacy_checkpoints",
        "normalize_model_name",
    }:
        from . import gat_utils

        mapping = {
            "GAT_MODEL_PRESETS": gat_utils.GAT_MODEL_PRESETS,
            "apply_checkpoint_args": gat_utils.apply_checkpoint_args,
            "get_checkpoint_state": gat_utils.get_checkpoint_state,
            "get_gat_config": gat_utils.get_gat_config,
            "load_encoders": gat_utils.load_encoders,
            "load_gat_generator_from_checkpoint": gat_utils.load_gat_generator_from_checkpoint,
            "load_gat_pipeline": gat_utils.load_gat_pipeline,
            "load_legacy_checkpoints": gat_utils.load_legacy_checkpoints,
            "normalize_model_name": gat_utils.normalize_model_name,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
