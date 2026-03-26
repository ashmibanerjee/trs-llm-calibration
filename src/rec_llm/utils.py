"""Utility functions for loading prompts and managing results."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_prompts(prompts_dir: Path) -> Tuple[str, str]:
    """
    Load system and user prompt templates from files.

    Args:
        prompts_dir: Path to the prompts directory

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    sys_prompt_file = prompts_dir / "sys_prompt.txt"
    usr_prompt_file = prompts_dir / "usr_prompt.txt"

    if not sys_prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found: {sys_prompt_file}")

    if not usr_prompt_file.exists():
        raise FileNotFoundError(f"User prompt file not found: {usr_prompt_file}")

    with open(sys_prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    with open(usr_prompt_file, 'r', encoding='utf-8') as f:
        user_prompt_template = f.read().strip()

    return system_prompt, user_prompt_template


def load_existing_results(output_file: Path) -> Tuple[List[Dict], set]:
    """
    Load existing results from output file if it exists.

    Args:
        output_file: Path to the output JSON file

    Returns:
        Tuple of (existing_results list, set of successfully processed query_ids)
    """
    if not output_file.exists():
        return [], set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Extract query IDs that have been successfully processed with recommendations
        processed_ids = set()
        for r in results:
            if 'query_id' not in r:
                continue

            # Only consider as processed if:
            # 1. success == True
            # 2. Has rec_cities data
            is_successful = r.get('success', False)
            has_recommendations = 'rec_cities' in r and r['rec_cities']

            if is_successful and has_recommendations:
                processed_ids.add(r['query_id'])

        print(f"Found {len(results)} existing results")
        print(f"Successfully processed query IDs: {len(processed_ids)}")
        if processed_ids:
            print(f"  Sample IDs: {sorted(list(processed_ids))[:5]}")

        return results, processed_ids

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse existing results file: {e}")
        print("Starting fresh...")
        return [], set()


def save_results(results: List[Dict], output_file: Path) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_file: Path to save the results
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def format_user_prompt(template: str, query: str) -> str:
    """
    Format user prompt template with the actual query.

    Args:
        template: User prompt template with {user_query} placeholder
        query: Actual user query

    Returns:
        Formatted prompt string
    """
    return template.replace("{user_query}", query)
