"""Utility functions for direct reasoner evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_recommendations(rec_file: Path) -> List[Dict]:
    """
    Load recommendation results from JSON file.

    Args:
        rec_file: Path to recommendations JSON file

    Returns:
        List of recommendation dictionaries
    """
    if not rec_file.exists():
        raise FileNotFoundError(f"Recommendations file not found: {rec_file}")

    with open(rec_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompts(prompts_dir: Path) -> Tuple[str, str]:
    """
    Load judge system and user prompt templates.

    Args:
        prompts_dir: Path to prompts directory

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    sys_prompt_file = prompts_dir / "sys_prompt.txt"
    usr_prompt_file = prompts_dir / "usr_prompt.txt"

    if not sys_prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {sys_prompt_file}")

    if not usr_prompt_file.exists():
        raise FileNotFoundError(f"User prompt not found: {usr_prompt_file}")

    with open(sys_prompt_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    with open(usr_prompt_file, 'r', encoding='utf-8') as f:
        user_prompt_template = f.read().strip()

    return system_prompt, user_prompt_template


def load_existing_evaluations(output_file: Path) -> Tuple[List[Dict], set]:
    """
    Load existing evaluation results.

    Args:
        output_file: Path to output JSON file

    Returns:
        Tuple of (existing_results, set of processed query_ids)
    """
    if not output_file.exists():
        return [], set()

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Extract query IDs that have been successfully evaluated
        processed_ids = set()

        for r in results:
            if 'query_id' not in r:
                continue

            # Check if the evaluation was successful
            if not r.get('success', False):
                # Skip failed evaluations - they will be retried
                continue

            # Check if we have a valid evaluation
            # Support both pairwise format (L1/L2) and itemwise format (itemwise_comparison)
            evaluation = r.get('evaluation')
            if not evaluation or not isinstance(evaluation, dict):
                continue

            # Pairwise format check (legacy: L1/L2 keys)
            has_pairwise = (
                'L1' in evaluation and
                'L2' in evaluation and
                isinstance(evaluation['L1'], dict) and
                isinstance(evaluation['L2'], dict)
            )

            # Newer pairwise structure used by Gemini evaluations
            has_pairwise_v2 = (
                'Pairwise_Comparisons' in evaluation and
                isinstance(evaluation['Pairwise_Comparisons'], dict)
            )

            # Itemwise format check
            has_itemwise = (
                'itemwise_comparison' in evaluation and
                isinstance(evaluation['itemwise_comparison'], dict)
            )

            # Consider as processed if we have a complete evaluation AND success=True
            if has_pairwise or has_pairwise_v2 or has_itemwise:
                processed_ids.add(r['query_id'])

        print(f"  Found {len(results)} existing evaluations")
        print(f"  Successfully processed: {len(processed_ids)}")
        if len(results) > len(processed_ids):
            print(f"  Failed evaluations to retry: {len(results) - len(processed_ids)}")

        # Return ALL results (including failed ones) so they are preserved,
        # but only successful query_ids are in processed_ids
        return results, processed_ids

    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse existing results: {e}")
        return [], set()


def save_evaluations(results: List[Dict], output_file: Path) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: List of evaluation result dictionaries
        output_file: Path to save results
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use json.dumps() and write the string to satisfy static type checkers
    json_text = json.dumps(results, indent=2, ensure_ascii=False)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_text)


def format_recommendation_list(rec_data: Dict) -> str:
    """
    Format a recommendation result into a readable string for the judge.

    Args:
        rec_data: Recommendation dictionary

    Returns:
        Formatted string representation
    """
    if not rec_data.get('success', False) or 'rec_cities' not in rec_data:
        return "No valid recommendations"

    cities = rec_data['rec_cities']
    formatted = []

    for i, city in enumerate(cities, 1):
        city_name = city.get('city', 'Unknown')
        country = city.get('country', 'Unknown')
        reason = city.get('reason', 'No reason provided')
        formatted.append(f"{i}. {city_name}, {country}\n   Reason: {reason}")

    return "\n".join(formatted)


def create_judge_prompt(
    user_prompt_template: str,
    query: str,
    rec_list: Dict,
    model_name: str
) -> str:
    """
    Create the judge evaluation prompt.

    Args:
        user_prompt_template: User prompt template with placeholders
        query: Original user query
        rec_list: Recommendation list to evaluate
        model_name: Name of the RecLLM model

    Returns:
        Formatted prompt string
    """
    formatted_list = format_recommendation_list(rec_list)

    # Replace placeholders
    prompt = user_prompt_template.replace("{query}", query)
    prompt = prompt.replace("{recommendation_list}", formatted_list)
    prompt = prompt.replace("{model_name}", model_name)

    return prompt


def create_combined_judge_prompt(
    user_prompt_template: str,
    query: str,
    rec_L1: Dict,
    rec_L2: Dict,
    model_L1: str,
    model_L2: str
) -> str:
    """
    Create the judge evaluation prompt comparing two recommendation lists.

    Args:
        user_prompt_template: User prompt template with placeholders
        query: Original user query
        rec_L1: First recommendation list
        rec_L2: Second recommendation list
        model_L1: Name of the L1 RecLLM model
        model_L2: Name of the L2 RecLLM model

    Returns:
        Formatted prompt string
    """
    formatted_L1 = format_recommendation_list(rec_L1)
    formatted_L2 = format_recommendation_list(rec_L2)

    # Replace placeholders
    prompt = user_prompt_template.replace("{query}", query)
    prompt = prompt.replace("L1: [L1 recommendations]", f"L1 ({model_L1}):\n{formatted_L1}")
    prompt = prompt.replace("L2: [L2 recommendations]", f"L2 ({model_L2}):\n{formatted_L2}")

    return prompt

def get_query_ids_for_human_eval(query_file):
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    query_ids = []
    for query in queries:
        query_ids.append(query['query_id'])
    return query_ids