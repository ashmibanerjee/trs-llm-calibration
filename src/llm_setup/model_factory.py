"""Model factory for creating LLM instances."""

from typing import Dict, Any, Type
from .base_model import BaseLLMModel
from .gemini_model import GeminiModel
from .gpt_model import GPTModel
from .claude_model import ClaudeModel
from .deepseek_model import DeepSeekModel
from .qwen_model import QwenModel


# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseLLMModel]] = {
    "gemini": GeminiModel,
    "gpt": GPTModel,
    "openai": GPTModel,  # Alias
    "claude": ClaudeModel,
    "anthropic": ClaudeModel,  # Alias
    "deepseek": DeepSeekModel,
    "qwen": QwenModel,
}


# Predefined model configurations
MODEL_CONFIGS = {
    # Gemini models
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "model_name": "gemini-1.5-pro-002",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model_name": "gemini-2.5-pro",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
    },
    "gemini-2.0-flash-exp": {
        "provider": "gemini",
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
    },

    # OpenAI models
    "gpt-4o": {
        "provider": "gpt",
        "model_name": "gpt-4o",
        "temperature": 0.0,  # Deterministic
        "max_completion_tokens": 8192,
    },
    "gpt-4o-mini": {
        "provider": "gpt",
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,  # Deterministic
        "max_completion_tokens": 8192,
    },
    "gpt-5": {
        "provider": "gpt",
        "model_name": "gpt-5",
        "reasoning_effort": "medium",
        # Note: gpt-5 (reasoning model) doesn't support temperature/top_p
    },
    "gpt-4-turbo": {
        "provider": "gpt",
        "model_name": "gpt-4-turbo-preview",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
    },

    # Claude models (direct API)
    "claude-3.5-sonnet": {
        "provider": "claude",
        "model_name": "claude-3-5-sonnet-20241022",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
    },
    "claude-3-opus": {
        "provider": "claude",
        "model_name": "claude-3-opus-20240229",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
    },

    # Claude 4.0 via Vertex AI
    "claude-4-sonnet": {
        "provider": "claude",
        "model_name": "claude-sonnet-4@20250514",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "use_vertex": True,
        "vertex_region": "us-east5",
    },

    # DeepSeek models (via Vertex AI)
    "deepseek-v3": {
        "provider": "deepseek",
        "model_name": "deepseek-ai/deepseek-v3.1-maas",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
    },

    # Qwen models (via Vertex AI)
    "qwen-3-next-80b": {
        "provider": "qwen",
        "model_name": "qwen/qwen3-next-80b-a3b-thinking-maas",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
    },
}


def get_model(model_identifier: str, **kwargs) -> BaseLLMModel:
    """
    Factory function to create a model instance.

    Args:
        model_identifier: Either a provider name (gemini/gpt/claude) or a
                         predefined model config key (gemini-2.0-flash-exp, gpt-4o, etc.)
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        Initialized model instance

    Examples:
        >>> # Use predefined configurations
        >>> gemini_model = get_model("gemini-2.0-flash-exp")
        >>> gpt_model = get_model("gpt-4o", temperature=0.5)
        >>> custom_model = get_model("gemini", model_name="gemini-1.5-pro")
        >>> # Use Claude 4.0 via Vertex AI
        >>> claude_vertex = get_model("claude-4-sonnet")
    """
    # Check if it's a predefined config
    if model_identifier in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_identifier].copy()
        provider = config.pop("provider")
        # Override with any user-provided kwargs
        config.update(kwargs)
    else:
        # Assume it's a provider name
        provider = model_identifier.lower()
        config = kwargs

    # Get the model class
    if provider not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model provider: {provider}. "
            f"Available providers: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[provider]

    # Create and initialize the model
    model = model_class(**config)
    model.initialize()

    return model


def list_available_models() -> Dict[str, Any]:
    """
    List all available model configurations.

    Returns:
        Dictionary of model identifiers and their configs
    """
    return MODEL_CONFIGS.copy()
