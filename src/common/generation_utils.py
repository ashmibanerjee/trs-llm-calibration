"""Shared utilities for generation pipelines."""

from pathlib import Path
from typing import Tuple, Callable, Any, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

def initialize_model(model_identifier: str, get_model_func: Callable, **kwargs) -> Tuple[Optional[Any], Optional[Dict]]:
    """
    Initialize a model with error handling.

    Args:
        model_identifier: Model identifier string
        get_model_func: Function to get the model (from llm_setup)
        **kwargs: Additional model configuration

    Returns:
        Tuple of (model_instance, model_info_dict) or (None, None) on error
    """
    print(f"  Initializing model: {model_identifier}...")
    try:
        model = get_model_func(model_identifier, **kwargs)
        model_info = model.get_model_info()
        print(f"  ✓ Model: {model_info['model_name']}")
        print(f"  ✓ Config: {model_info['config']}")
        return model, model_info
    except Exception as e:
        print(f"  ✗ Error initializing model: {e}")
        return None, None


def load_prompts_with_logging(prompts_dir: Path, load_func: Callable) -> Optional[Tuple]:
    """
    Load prompts with logging.

    Args:
        prompts_dir: Path to prompts directory
        load_func: Function to load prompts

    Returns:
        Tuple of prompts or None on error
    """
    print(f"  Loading prompts from: {prompts_dir}")
    try:
        prompts = load_func(prompts_dir)
        system_prompt = prompts[0]
        user_prompt_template = prompts[1]
        print(f"  ✓ System prompt: {len(system_prompt)} characters")
        print(f"  ✓ User prompt template: {len(user_prompt_template)} characters")
        return prompts
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return None


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print("=" * width)
    print(title)
    print("=" * width)


def print_step(step_num: int, total_steps: int, message: str):
    """Print a step message."""
    print(f"\n[{step_num}/{total_steps}] {message}")


def print_summary(results: list, result_key: str = 'success', extra_stats: dict = None):
    """
    Print a summary of results.

    Args:
        results: List of result dictionaries
        result_key: Key to check for success (default: 'success')
        extra_stats: Optional dict of additional statistics to display
    """
    successful = sum(1 for r in results if r.get(result_key, False))
    failed = len(results) - successful

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if len(results) > 0:
        print(f"Success rate: {successful/len(results)*100:.1f}%")

    if extra_stats:
        for key, value in extra_stats.items():
            if isinstance(value, int) and value > 1000:
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")


def check_all_processed(to_process: list, output_location: Path) -> bool:
    """
    Check if all items are already processed.

    Args:
        to_process: List of items to process
        output_location: Path where results are saved

    Returns:
        True if nothing left to process
    """
    if not to_process:
        print("  ✓ All items already processed!")
        print(f"\n  Results at: {output_location}")
        return True
    return False
