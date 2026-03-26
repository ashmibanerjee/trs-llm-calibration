"""LLM setup package."""

from .model_factory import get_model, list_available_models, MODEL_CONFIGS

__all__ = [
    'get_model',
    'list_available_models',
    'MODEL_CONFIGS'
]
