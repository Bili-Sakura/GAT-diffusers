from typing import Any, Dict


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
