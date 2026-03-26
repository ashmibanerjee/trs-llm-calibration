"""Base class for all LLM models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMModel(ABC):
    """Abstract base class for all LLM model implementations."""

    def __init__(self, model_name: str, **config):
        """
        Initialize the model.

        Args:
            model_name: Name/identifier of the model
            **config: Model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response from the model.

        Args:
            system_prompt: System instruction/context
            user_prompt: User query/prompt

        Returns:
            Generated response as string
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model and any required authentication/setup."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.

        Returns:
            Dictionary with model name and config
        """
        return {
            "model_name": self.model_name,
            "config": self.config
        }
