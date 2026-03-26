"""Configuration for recommendation generation."""

from pathlib import Path


class ModelConfig:
    """Model configurations for different LLMs."""

    # Available model configurations
    GEMINI_FLASH = {
        "model_identifier": "gemini-2.5-flash",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
        "top_p": 1.0,
    }

    GEMINI_2_0_FLASH = {
        "model_identifier": "gemini-2.0-flash-exp",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
        "top_p": 1.0,
    }

    GEMINI_PRO = {
        "model_identifier": "gemini-1.5-pro",
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,
        "top_p": 1.0,
    }

    GPT_4O = {
        "model_identifier": "gpt-4o",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    GPT_4O_MINI = {
        "model_identifier": "gpt-4o-mini",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    GPT_4_TURBO = {
        "model_identifier": "gpt-4-turbo",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    CLAUDE_SONNET = {
        "model_identifier": "claude-3.5-sonnet",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    CLAUDE_OPUS = {
        "model_identifier": "claude-3-opus",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    # Claude 4.0 Sonnet via Vertex AI
    CLAUDE_4_SONNET = {
        "model_identifier": "claude-4-sonnet",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }

    # Qwen via Vertex AI
    QWEN_3_NEXT_80B = {
        "model_identifier": "qwen-3-next-80b",
        "temperature": 0.0,  # Deterministic
        "max_tokens": 8192,
        "top_p": 1.0,
    }


class PathConfig:
    """Path configuration for data and outputs."""

    # Use filtered queries JSON instead of CSV
    QUERIES_JSON = "data/conv-trs/ecir-2026/selected_queries/filtered_queries.json"
    QUERIES_CSV = "data/conv-trs/ecir-2026/selected_queries/Gemini1Point5Pro_all_queries.csv"  # Legacy
    OUTPUT_DIR = "data/conv-trs/ecir-2026/rec-llm"
    PROMPTS_DIR = "prompts/rec-llm"

    @staticmethod
    def get_output_filename(model_name: str) -> str:
        """
        Generate output filename based on model name.

        Args:
            model_name: Name of the model

        Returns:
            Output filename
        """
        # Clean model name for filename
        clean_name = model_name.replace(".", "_").replace("-", "_").replace("@", "_")
        return f"{clean_name}_recommendations.json"


class GenerationConfig:
    """Generation parameters."""

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    # Which model to use (change this to switch models)
    ACTIVE_MODEL = ModelConfig.CLAUDE_4_SONNET

    # Or use a different model:
    # ACTIVE_MODEL = ModelConfig.GEMINI_FLASH
    # ACTIVE_MODEL = ModelConfig.GPT_4O
    # ACTIVE_MODEL = ModelConfig.CLAUDE_SONNET
