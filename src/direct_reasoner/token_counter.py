from typing import Dict
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: Text to count tokens for
        model: Model name to use for encoding (default: gpt-4)

    Returns:
        Number of tokens
    """
    try:
        # Get the encoding for the model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_for_model(text: str, model_name: str) -> int:
    """
    Count tokens using the appropriate encoding for the model.

    Args:
        text: Text to count tokens for
        model_name: Full model name (e.g., "gemini-2.5-flash", "gpt-4o")

    Returns:
        Number of tokens
    """
    # Map model names to tiktoken encodings
    if "gpt-4" in model_name or "gpt-3.5" in model_name:
        return count_tokens(text, "gpt-4")
    elif "gemini" in model_name or "claude" in model_name:
        # Use cl100k_base as approximation for Gemini and Claude
        return count_tokens(text, "gpt-4")
    else:
        # Default to cl100k_base
        return count_tokens(text, "gpt-4")


def get_token_stats(input_text: str, output_text: str, model_name: str) -> Dict[str, int]:
    """
    Get token statistics for input and output.

    Args:
        input_text: Input text (prompt)
        output_text: Output text (response)
        model_name: Model name

    Returns:
        Dictionary with token counts
    """
    input_tokens = count_tokens_for_model(input_text, model_name)
    output_tokens = count_tokens_for_model(output_text, model_name)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }
"""Token counting utilities using tiktoken."""

import tiktoken
from typing import Dict

