"""Anthropic Claude model implementation with support for both direct API and Vertex AI."""

import os
from anthropic import Anthropic
from anthropic import AnthropicVertex

from .base_model import BaseLLMModel


class ClaudeModel(BaseLLMModel):
    """Anthropic Claude model supporting both direct API and Vertex AI."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        top_p: float = 0.95,
        use_vertex: bool = False,
        vertex_region: str = "us-east5",
        vertex_project_id: str = None,
        **kwargs
    ):
        """
        Initialize Claude model.

        Args:
            model_name: Claude model name
            api_key: Anthropic API key (for direct API)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            use_vertex: Whether to use Vertex AI instead of direct API
            vertex_region: Vertex AI region (default: us-east5)
            vertex_project_id: Google Cloud project ID for Vertex AI
        """
        super().__init__(model_name=model_name)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.use_vertex = use_vertex
        self.vertex_region = vertex_region
        self.vertex_project_id = vertex_project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        self.config.update({
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "use_vertex": use_vertex,
            "vertex_region": vertex_region if use_vertex else None,
        })

        self._client = None

    def initialize(self) -> None:
        """Initialize Anthropic client (either direct or Vertex AI)."""
        if self.use_vertex:
            # Initialize Vertex AI client
            if not self.vertex_project_id:
                raise ValueError(
                    "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT env variable for Vertex AI"
                )

            self._client = AnthropicVertex(
                region=self.vertex_region,
                project_id=self.vertex_project_id
            )
            print(f"✓ Initialized Claude model via Vertex AI: {self.model_name}")
            print(f"  Project: {self.vertex_project_id}")
            print(f"  Region: {self.vertex_region}")
        else:
            # Initialize direct API client
            if not self.api_key:
                raise ValueError(
                    "API key must be provided or set in ANTHROPIC_API_KEY env variable"
                )

            self._client = Anthropic(api_key=self.api_key)
            print(f"✓ Initialized Claude model: {self.model_name}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from Claude model.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            Generated text response
        """
        if self._client is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "provider": "claude" + ("-vertex" if self.use_vertex else ""),
            "model_name": self.model_name,
            "config": self.config
        }
