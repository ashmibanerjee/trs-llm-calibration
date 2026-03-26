"""Generate city recommendations for queries using any LLM via modular interface."""

import pandas as pd
import json
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_setup import get_model
from src.rec_llm.utils import load_prompts, load_existing_results, save_results, format_user_prompt
from src.rec_llm.response_handler import generate_recommendation
from src.rec_llm.config import PathConfig, GenerationConfig, ModelConfig
from src.common.generation_utils import (
    initialize_model,
    load_prompts_with_logging,
    print_header,
    print_step,
    print_summary,
    check_all_processed
)


def load_queries_from_json(json_path: Path):
    """
    Load queries from JSON file.

    Args:
        json_path: Path to JSON file containing queries

    Returns:
        List of tuples (query_id, query_text)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Expected format: [{"query_id": "...", "query_text": "..."}, ...]
    queries = []
    for item in data:
        query_id = item.get('query_id')
        query_text = item.get('query_text')
        if query_id and query_text:
            queries.append((query_id, query_text))

    return queries


def get_query_column(df: pd.DataFrame) -> str:
    """
    Identify the query column in the DataFrame.

    Args:
        df: DataFrame containing queries

    Returns:
        Column name containing queries

    Raises:
        ValueError: If no query column found
    """
    for col in ['query', 'Query', 'text', 'question']:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find query column. Available columns: {df.columns.tolist()}"
    )


def get_config_id_column(df: pd.DataFrame):
    """
    Identify the config_id column in the DataFrame.

    Args:
        df: DataFrame containing queries

    Returns:
        Column name containing config_id, or None if not found
    """
    for col in ['config_id', 'Config_ID', 'id', 'ID']:
        if col in df.columns:
            return col
    return None


def main(model_config: dict = None, use_json: bool = True):
    """
    Main function to process all queries.

    Args:
        model_config: Optional model configuration dictionary.
                     If None, uses GenerationConfig.ACTIVE_MODEL
        use_json: If True, use JSON file (filtered_queries.json), else use CSV
    """
    # Use provided config or default from GenerationConfig
    if model_config is None:
        model_config = GenerationConfig.ACTIVE_MODEL

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    prompts_dir = base_dir / PathConfig.PROMPTS_DIR
    output_dir = base_dir / PathConfig.OUTPUT_DIR

    # Get model identifier for output filename
    model_identifier = model_config.get("model_identifier", "unknown_model")

    # Create a copy of model_config without model_identifier for passing to initialize_model
    model_kwargs = {k: v for k, v in model_config.items() if k != "model_identifier"}

    output_file = output_dir / PathConfig.get_output_filename(model_identifier)

    print_header("City Recommendation Generator")

    # Initialize model using factory
    print_step(1, 5, f"Initializing model: {model_identifier}")
    model, model_info = initialize_model(model_identifier, get_model, **model_kwargs)
    if model is None:
        return

    # Load prompts
    print_step(2, 5, "Loading prompts")
    prompts = load_prompts_with_logging(prompts_dir, load_prompts)
    if prompts is None:
        return
    system_prompt, user_prompt_template = prompts

    # Load queries
    print_step(3, 5, "Loading queries")

    queries_to_load = []

    if use_json:
        # Load from JSON file
        queries_json = base_dir / PathConfig.QUERIES_JSON
        print(f"  File: {queries_json}")
        if not queries_json.exists():
            print(f"  ✗ ERROR: Queries file not found: {queries_json}")
            return

        queries_to_load = load_queries_from_json(queries_json)
        print(f"  ✓ Loaded {len(queries_to_load)} queries from JSON")
    else:
        # Load from CSV file (legacy)
        queries_csv = base_dir / PathConfig.QUERIES_CSV
        print(f"  File: {queries_csv}")
        if not queries_csv.exists():
            print(f"  ✗ ERROR: Queries file not found: {queries_csv}")
            return

        df = pd.read_csv(queries_csv)
        print(f"  ✓ Loaded {len(df)} queries from CSV")

        try:
            query_column = get_query_column(df)
            print(f"  ✓ Using column: '{query_column}'")
        except ValueError as e:
            print(f"  ✗ {e}")
            return

        config_id_column = get_config_id_column(df)
        for idx, row in df.iterrows():
            # Use config_id if available, otherwise use row index
            query_id = row[config_id_column] if config_id_column else idx
            queries_to_load.append((query_id, row[query_column]))

    # Load existing results
    print_step(4, 5, "Checking for existing results")
    results, processed_ids = load_existing_results(output_file)

    if processed_ids:
        print(f"  ✓ Found {len(processed_ids)} already processed queries")
        print(f"  → Will skip these and process remaining {len(queries_to_load) - len(processed_ids)} queries")
    else:
        print(f"  → No existing results found, will process all {len(queries_to_load)} queries")

    # Generate recommendations
    print_step(5, 5, "Generating recommendations")
    print(f"  Output: {output_file}\n")

    queries_to_process = []
    for query_id, query_text in queries_to_load:
        if query_id not in processed_ids:
            queries_to_process.append((query_id, query_text))

    if check_all_processed(queries_to_process, output_file):
        return

    # Process queries with progress bar
    for query_id, query in tqdm(queries_to_process, desc="Processing queries"):
        # Format user prompt
        user_prompt = format_user_prompt(user_prompt_template, query)

        # Generate recommendation with retry on failure
        result = None
        for attempt in range(1, GenerationConfig.MAX_RETRIES + 1):
            result = generate_recommendation(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                query=query,
                max_retries=GenerationConfig.MAX_RETRIES,
                retry_delay=GenerationConfig.RETRY_DELAY
            )

            # If successful, break out of retry loop
            if result.get('success', False):
                break

            # If failed and not the last attempt, sleep and retry
            if attempt < GenerationConfig.MAX_RETRIES:
                print(f"  ⚠ Query {query_id} failed (attempt {attempt}/{GenerationConfig.MAX_RETRIES})")
                print(f"  → Sleeping for {GenerationConfig.RETRY_DELAY}s before retry...")
                import time
                time.sleep(GenerationConfig.RETRY_DELAY)
            else:
                print(f"  ✗ Query {query_id} failed after {GenerationConfig.MAX_RETRIES} attempts")

        # Add metadata
        result['query_id'] = str(query_id)
        result['model'] = model_info['model_name']
        result['model_config'] = model_info['config']

        # Append to results
        results.append(result)

        # Save after each query (resume capability)
        save_results(results, output_file)

    # Print summary
    print_summary(results, extra_stats={'Output file': str(output_file)})


if __name__ == "__main__":
    # Run with default model from config (Claude 4.0 Sonnet via Vertex AI)
    # and use filtered_queries.json
    # main(model_config=ModelConfig.CLAUDE_4_SONNET, use_json=True)
    main(model_config=ModelConfig.QWEN_3_NEXT_80B, use_json=True)
    # Or run with a specific model:
    # from src.rec_llm.config import ModelConfig
    # main(model_config=ModelConfig.GPT_4O, use_json=True)
    # main(model_config=ModelConfig.GEMINI_FLASH, use_json=True)
