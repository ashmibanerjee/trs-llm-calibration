"""OpenAI GPT model implementation."""
import os
from openai import OpenAI
from .base_model import BaseLLMModel


class GPTModel(BaseLLMModel):
    """OpenAI GPT model."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = None,
        max_completion_tokens: int = None,
        top_p: float = 1.0,
        reasoning_effort: str = None,
        **kwargs
    ):
        """
        Initialize GPT model.

        Args:
            model_name: GPT model name
            api_key: OpenAI API key
            temperature: Sampling temperature (not supported for o1/o3/gpt-5 models)
            max_tokens: Maximum tokens to generate (for older models)
            max_completion_tokens: Maximum completion tokens (for newer models)
            top_p: Top-p sampling parameter (not supported for o1/o3/gpt-5 models)
            reasoning_effort: Reasoning effort for o1/o3/gpt-5 models
        """
        super().__init__(model_name=model_name, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

        # Detect if this is a reasoning model (o1, o3, gpt-5)
        self.is_reasoning_model = any(
            model_name.startswith(prefix)
            for prefix in ['o1', 'o3', 'gpt-5']
        )

        self.config.update({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "top_p": top_p,
        })

        if reasoning_effort:
            self.config["reasoning_effort"] = reasoning_effort

        self._client = None

    def initialize(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENAI_API_KEY env variable"
            )

        self._client = OpenAI(api_key=self.api_key)
        print(f"✓ Initialized GPT model: {self.model_name}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from GPT model.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            Generated text response
        """
        if self._client is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Build kwargs for API call
        api_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        # Reasoning models (o1, o3, gpt-5) don't support temperature/top_p
        if not self.is_reasoning_model:
            api_kwargs["temperature"] = self.temperature
            api_kwargs["top_p"] = self.top_p

        # Add token limit parameters based on model
        if self.max_completion_tokens:
            api_kwargs["max_completion_tokens"] = self.max_completion_tokens
        elif self.max_tokens:
            api_kwargs["max_tokens"] = self.max_tokens

        # Add reasoning parameters for o1/o3/gpt-5 models if specified
        if self.reasoning_effort:
            api_kwargs["reasoning_effort"] = self.reasoning_effort

        response = self._client.chat.completions.create(**api_kwargs)

        return response.choices[0].message.content

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "config": self.config
        }
