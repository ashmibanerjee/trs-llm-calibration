"""Qwen model implementation via Vertex AI."""

import os
import json
import base64
from typing import Optional
import vertexai
from google.auth.transport.requests import Request
import requests
from dotenv import load_dotenv
from .base_model import BaseLLMModel
import google.auth
from google.auth.transport.requests import Request as AuthRequest

# Load environment variables from .env file
load_dotenv()


def _decode_service_key():
    """Decode base64-encoded service key from environment variable."""
    encoded_key = os.environ.get("GOOGLE_CREDENTIALS")
    if not encoded_key:
        return None
    original_service_key = json.loads(base64.b64decode(encoded_key).decode('utf-8'))
    if original_service_key:
        return original_service_key
    return None


class QwenModel(BaseLLMModel):
    """Qwen model implementation using Vertex AI Publisher Model API."""

    def __init__(
        self,
        model_name: str = "qwen/qwen3-next-80b-a3b-thinking-maas",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        project_id: Optional[str] = None,
        region: str = "global",
        **kwargs
    ):
        """Initialize Qwen model via Vertex AI."""
        super().__init__(model_name=model_name, **kwargs)

        self.temperature = temperature
        self.max_tokens = max_tokens or 8192
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "genai-experiments-397219")
        self.region = region
        self._access_token = None
        self._endpoint_url = None

    def initialize(self) -> None:
        """Initialize the Qwen model via Vertex AI."""
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)

        # Get credentials from environment variable
        _decode_service_key()
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        auth_request = AuthRequest()
        credentials.refresh(auth_request)
        access_token = credentials.token
        self._access_token = access_token

        # Construct the REST API endpoint URL for Qwen
        self._endpoint_url = (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"endpoints/openapi/chat/completions"
        )

        print(f"✓ Qwen model initialized")
        print(f"  Project: {self.project_id}")
        print(f"  Region: {self.region}")
        print(f"  Model: {self.model_name}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the Qwen model."""
        if not self._access_token:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Prepare the request
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        request_payload = {
            "model": self.model_name,
            "stream": False,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Make the API request
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            self._endpoint_url,
            headers=headers,
            data=json.dumps(request_payload),
            timeout=60
        )

        # Check for errors
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")

        # Parse and extract content
        response_json = response.json()

        if "choices" in response_json and response_json["choices"]:
            for choice in response_json["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]

        raise ValueError(f"Could not extract content from response: {response_json}")

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "provider": "qwen",
            "model_name": self.model_name,
            "config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "project_id": self.project_id,
                "region": self.region
            }
        }

