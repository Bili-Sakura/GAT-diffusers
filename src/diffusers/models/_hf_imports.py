"""Lazy imports from the installed Hugging Face diffusers package."""

from __future__ import annotations

from .._hf import get_hf_attr


def get_config_mixin():
    return get_hf_attr("diffusers.configuration_utils.ConfigMixin")


def get_register_to_config():
    return get_hf_attr("diffusers.configuration_utils.register_to_config")


def get_model_mixin():
    return get_hf_attr("diffusers.models.modeling_utils.ModelMixin")


def get_base_output():
    return get_hf_attr("diffusers.utils.BaseOutput")
