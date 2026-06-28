"""Load symbols from the installed Hugging Face diffusers package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

_LOCAL_ROOT = Path(__file__).resolve().parent
_LOCAL_SRC = _LOCAL_ROOT.parent
_CACHED: Optional[object] = None


def _without_local_src_path() -> list[str]:
    return [entry for entry in sys.path if Path(entry).resolve() != _LOCAL_SRC.resolve()]


def _is_local_diffusers_module(module_name: str) -> bool:
    module = sys.modules.get(module_name)
    if module is None:
        return False
    module_file = getattr(module, "__file__", "") or ""
    if str(_LOCAL_ROOT) in module_file or str(_LOCAL_SRC) in module_file:
        return True
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        try:
            return any(str(_LOCAL_ROOT) in str(path) for path in module_paths)
        except (KeyError, ValueError, TypeError):
            return str(_LOCAL_ROOT) in module_file
    return False


def _stash_local_diffusers_modules() -> dict[str, Any]:
    stashed: dict[str, Any] = {}
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            if _is_local_diffusers_module(module_name):
                stashed[module_name] = sys.modules.pop(module_name)
    return stashed


def _restore_modules(stashed: dict[str, Any]) -> None:
    for module_name, module in stashed.items():
        sys.modules[module_name] = module


def get_hf_diffusers():
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    stashed = _stash_local_diffusers_modules()
    original_path = sys.path[:]
    try:
        sys.path = _without_local_src_path()
        _CACHED = importlib.import_module("diffusers")
    finally:
        sys.path = original_path
        _restore_modules(stashed)

    return _CACHED


def get_hf_attr(dotted_path: str):
    module_path, _, attr_name = dotted_path.rpartition(".")
    stashed = _stash_local_diffusers_modules()
    original_path = sys.path[:]
    try:
        sys.path = _without_local_src_path()
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    finally:
        sys.path = original_path
        _restore_modules(stashed)


def get_config_mixin():
    return get_hf_attr("diffusers.configuration_utils.ConfigMixin")


def get_register_to_config():
    return get_hf_attr("diffusers.configuration_utils.register_to_config")


def get_model_mixin():
    return get_hf_attr("diffusers.models.modeling_utils.ModelMixin")


def get_base_output():
    return get_hf_attr("diffusers.utils.BaseOutput")
