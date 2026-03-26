"""Google Gemini model implementation."""

import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from .base_model import BaseLLMModel


class GeminiModel(BaseLLMModel):
    """Google Gemini model via Vertex AI."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        top_p: float = 1.0,
        top_k: int = 40,
        project_id: str = None,
        location: str = "us-central1",
        **kwargs
    ):
        """
        Initialize Gemini model.

        Args:
            model_name: Gemini model name
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            project_id: Google Cloud project ID
            location: Vertex AI location
        """
        super().__init__(model_name=model_name, **kwargs)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location

        self.config.update({
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
        })

        self._model = None

    def initialize(self) -> None:
        """Initialize Vertex AI and Gemini model."""
        if not self.project_id:
            raise ValueError(
                "Project ID must be provided or set in GOOGLE_CLOUD_PROJECT env variable"
            )

        vertexai.init(project=self.project_id, location=self.location)

        generation_config = GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        self._model = GenerativeModel(
            self.model_name,
            generation_config=generation_config,
        )

        print(f"✓ Initialized Gemini model: {self.model_name}")
        print(f"  Project: {self.project_id}")
        print(f"  Location: {self.location}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from Gemini model.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            Generated text response
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Combine system and user prompts (Gemini doesn't separate them in the same way)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self._model.generate_content(full_prompt)
        return response.text

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "config": self.config
        }

