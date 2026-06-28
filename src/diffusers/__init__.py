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
    "load_legacy_checkpoints",
    "normalize_model_name",
    "save_gat_pipeline_pretrained",
]


def __getattr__(name: str):
    if name == "GATPipeline":
        from .pipelines.gat.gat import GATPipeline

        return GATPipeline
    if name == "save_gat_pipeline_pretrained":
        from .pipelines.gat.gat import save_gat_pipeline_pretrained

        return save_gat_pipeline_pretrained
    if name in {
        "GATGenerator",
        "GAT_models",
        "GATDiscriminator",
        "GATD_models",
        "GAT_MODEL_PRESETS",
        "RpGANLoss",
        "RpGANPTLoss",
        "apply_checkpoint_args",
        "convert_gat_checkpoint",
        "get_checkpoint_state",
        "get_gat_config",
        "load_encoders",
        "load_gat_generator_from_checkpoint",
        "load_legacy_checkpoints",
        "normalize_model_name",
    }:
        from .models.gat import gat as gat_module

        return getattr(gat_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
