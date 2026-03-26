"""Main evaluation script for direct reasoner (LLM-as-Judge)."""

import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.llm_setup import get_model
from src.direct_reasoner.config import ExperimentConfig, PathConfig, JudgeConfig
from src.direct_reasoner.utils import (
    load_recommendations,
    load_prompts,
    load_existing_evaluations,
    save_evaluations,
    create_combined_judge_prompt,
    get_query_ids_for_human_eval
)
from src.direct_reasoner.evaluator import evaluate_recommendations
from src.common.generation_utils import (
    initialize_model,
    load_prompts_with_logging,
    print_header,
    print_step,
    check_all_processed
)


# --- Helpers to improve modularity -------------------------------------------------

def load_selected_queries_if_requested(exp_config):
    """Return (selected_list, selected_set) or (None, None) if not requested."""
    if not exp_config.get('use_selected_queries'):
        return None, None

    try:
        selected_list = get_query_ids_for_human_eval(PathConfig.SELECTED_QUERIES_FILE)
        return selected_list, set(selected_list)
    except FileNotFoundError:
        print(f"  ✗ Selected queries file not found: {PathConfig.SELECTED_QUERIES_FILE}")
        return None, None


def determine_prompts_dir(exp_config):
    """Return a Path for prompts directory, allowing experiment override."""
    if exp_config.get('prompts_dir'):
        return PathConfig.BASE_DIR / 'prompts' / exp_config['prompts_dir']
    return PathConfig.PROMPTS_DIR


def get_effective_max_queries(cli_max, exp_config):
    """Return the integer max_queries to use (or None). CLI overrides experiment config."""
    if cli_max is not None:
        return cli_max
    cfg = exp_config.get('max_queries')
    if cfg is None:
        return None
    try:
        return int(cfg)
    except Exception:
        print(f"  ⚠ Warning: could not parse experiment max_queries='{cfg}' to int; ignoring limit")
        return None


def filter_and_prepare_recs(recommendations_L1, recommendations_L2):
    """Apply success==True filter if present, return filtered lists."""
    l1_f = [r for r in recommendations_L1 if r.get('success') is True]
    l2_f = [r for r in recommendations_L2 if r.get('success') is True]

    # If no items had explicit success flags, fall back to original lists
    if not l1_f:
        l1_f = recommendations_L1
    if not l2_f:
        l2_f = recommendations_L2

    return l1_f, l2_f


def build_matched_pairs(recommendations_L1_filtered, recommendations_L2_filtered, selected_list=None, selected_set=None):
    """Return matched_pairs between L1 and L2. If selected_list provided, preserve its order and restrict to it."""
    # Deduplicate L1 by query_id (keep first occurrence)
    l1_by_id = {}
    for r in recommendations_L1_filtered:
        qid = r.get('query_id')
        if qid and qid not in l1_by_id:
            l1_by_id[qid] = r

    l2_by_id = {r.get('query_id'): r for r in recommendations_L2_filtered if r.get('query_id')}

    matched = []
    # Helper to create a safe placeholder when a recommendation is missing
    def _placeholder(qid):
        return { 'query_id': qid, 'success': False }

    if selected_list is not None:
        missing_l1 = 0
        missing_l2 = 0
        for qid in selected_list:
            rec1 = l1_by_id.get(qid) or _placeholder(qid)
            rec2 = l2_by_id.get(qid) or _placeholder(qid)
            if rec1.get('success') is not True:
                missing_l1 += 1
            if rec2.get('success') is not True:
                missing_l2 += 1

            matched.append({
                'query_id': qid,
                'query': rec1.get('query', '') if rec1.get('query') else (rec2.get('query', '') if rec2.get('query') else ''),
                'rec_L1': rec1,
                'rec_L2': rec2
            })

        # Inform about missing recommendations so the user can inspect
        if missing_l1 or missing_l2:
            print(f"  ⚠ Note: {missing_l1} selected queries missing L1 recs, {missing_l2} missing L2 recs (placeholders will be used)")
    else:
        # union of keys from both L1 and L2 (ensure we evaluate all available pairs)
        all_qids = list(dict.fromkeys(list(l1_by_id.keys()) + list(l2_by_id.keys())))
        for qid in all_qids:
            rec1 = l1_by_id.get(qid) or _placeholder(qid)
            rec2 = l2_by_id.get(qid) or _placeholder(qid)
            matched.append({
                'query_id': qid,
                'query': rec1.get('query', '') if rec1.get('query') else (rec2.get('query', '') if rec2.get('query') else ''),
                'rec_L1': rec1,
                'rec_L2': rec2
            })

    return matched


def init_judge_model(judge_model_identifier):
    """Initialize judge model and return (judge_model, judge_info) or (None, None) on failure."""
    model_kwargs = {"temperature": JudgeConfig.TEMPERATURE}

    if "gpt-4o" in judge_model_identifier:
        model_kwargs["max_completion_tokens"] = 8192

    if "gpt-5" in judge_model_identifier:
        # GPT-5 is a reasoning model - only reasoning_effort is supported
        model_kwargs["reasoning_effort"] = "medium"

    if "deepseek" in judge_model_identifier:
        model_kwargs["max_tokens"] = 8192

    return initialize_model(judge_model_identifier, get_model, **model_kwargs)


# --- Main experimental runner ----------------------------------------------------

def run_experiment(experiment_name: str, max_queries: int = None, rerun_failed: bool = False):
    """Run a direct reasoner evaluation experiment comparing two recommendation sets."""
    print_header(f"Direct Reasoner Evaluation: {experiment_name}")

    exp_config = ExperimentConfig.get_experiment(experiment_name)
    judge_model_identifier = exp_config['judge_model']
    rec_file_L1_name = exp_config['rec_file_L1']
    rec_file_L2_name = exp_config['rec_file_L2']
    output_file_name = exp_config['output_file']

    print(f"\nConfiguration:")
    print(f"  Judge Model: {judge_model_identifier}")
    print(f"  Recommendations L1: {rec_file_L1_name}")
    print(f"  Recommendations L2: {rec_file_L2_name}")
    print(f"  Output: {output_file_name}")

    # determine effective max_queries
    effective_max = get_effective_max_queries(max_queries, exp_config)
    if effective_max:
        print(f"  Max Queries (limit): {effective_max}")

    # set up paths
    rec_file_L1 = PathConfig.get_rec_file(rec_file_L1_name)
    rec_file_L2 = PathConfig.get_rec_file(rec_file_L2_name)
    output_file = PathConfig.get_output_file(output_file_name)

    prompts_dir = determine_prompts_dir(exp_config)

    # load selected queries if requested
    selected_list, selected_set = load_selected_queries_if_requested(exp_config)
    if exp_config.get('use_selected_queries') and (selected_list is None or selected_set is None):
        # error already printed by helper
        return

    # initialize judge model
    print_step(1, 6, f"Initializing judge model: {judge_model_identifier}")
    judge_model, judge_info = init_judge_model(judge_model_identifier)
    if judge_model is None:
        return

    # load prompts
    print_step(2, 6, "Loading judge prompts")
    prompts = load_prompts_with_logging(prompts_dir, load_prompts)
    if prompts is None:
        return
    system_prompt, user_prompt_template = prompts

    # load recommendations
    print_step(3, 6, "Loading recommendations L1")
    print(f"  File: {rec_file_L1}")
    try:
        recommendations_L1 = load_recommendations(rec_file_L1)
        print(f"  ✓ Loaded {len(recommendations_L1)} recommendations from L1")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return

    print_step(4, 6, "Loading recommendations L2")
    print(f"  File: {rec_file_L2}")
    try:
        recommendations_L2 = load_recommendations(rec_file_L2)
        print(f"  ✓ Loaded {len(recommendations_L2)} recommendations from L2")
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return

    # filter by success flags if present
    recs_l1_filtered, recs_l2_filtered = filter_and_prepare_recs(recommendations_L1, recommendations_L2)
    print(f"  ✓ Using {len(recs_l1_filtered)} L1 recs and {len(recs_l2_filtered)} L2 recs after success-filtering")

    # when selected queries requested, further restrict lists (the build_matched_pairs will also respect selection)
    if selected_set is not None:
        recs_l1_filtered = [r for r in recs_l1_filtered if r.get('query_id') in selected_set]
        recs_l2_filtered = [r for r in recs_l2_filtered if r.get('query_id') in selected_set]

    # build matched pairs (preserving selected order if provided)
    matched_pairs = build_matched_pairs(recs_l1_filtered, recs_l2_filtered, selected_list, selected_set)
    if selected_list is not None:
        print(f"  ✓ Matched {len(matched_pairs)} query pairs after applying selected_queries filter (selected set size={len(selected_list)})")
    else:
        print(f"  ✓ Matched {len(matched_pairs)} query pairs")

    # load existing evaluations and compute remaining
    print_step(5, 6, "Checking for existing evaluations")
    evaluations, processed_ids = load_existing_evaluations(output_file)

    if processed_ids:
        print(f"  ✓ Found {len(processed_ids)} already evaluated queries")
        print(f"  → Will evaluate remaining {len(matched_pairs) - len(processed_ids)} queries")
    else:
        print(f"  → No existing evaluations, will evaluate all {len(matched_pairs)} queries")

    # If user requested to rerun failed evaluations, compute failed ids from existing results
    failed_ids = set()
    if rerun_failed:
        if not evaluations:
            print("  ✗ No existing evaluation file found to derive failed queries from; nothing to rerun.")
            return
        failed_ids = {r['query_id'] for r in evaluations if r.get('query_id') and not r.get('success', False)}
        if not failed_ids:
            print("  ✓ No failed evaluations found to rerun.")
            return
        print(f"  ✓ Found {len(failed_ids)} failed evaluations to rerun")

    # prepare to evaluate
    print_step(6, 6, "Evaluating recommendations")
    print(f"  Output: {output_file}\n")

    if rerun_failed:
        # Restrict to only previously failed query_ids (preserve order from matched_pairs)
        to_evaluate = [pair for pair in matched_pairs if pair['query_id'] in failed_ids]
    else:
        to_evaluate = [pair for pair in matched_pairs if pair['query_id'] not in processed_ids]

    # apply max_queries limit if provided
    if effective_max is not None and len(to_evaluate) > effective_max:
        print(f"  ⚠ Limiting evaluation to first {effective_max} queries (out of {len(to_evaluate)} remaining)")
        to_evaluate = to_evaluate[:effective_max]

    if check_all_processed(to_evaluate, output_file):
        return

    # evaluate each pair
    for pair in tqdm(to_evaluate, desc="Evaluating"):
        query_id = pair['query_id']
        query = pair['query']
        rec_L1 = pair['rec_L1']
        rec_L2 = pair['rec_L2']

        model_L1 = rec_L1.get('model_name', rec_file_L1_name.replace('_recommendations.json', ''))
        model_L2 = rec_L2.get('model_name', rec_file_L2_name.replace('_recommendations.json', ''))

        user_prompt = create_combined_judge_prompt(
            user_prompt_template,
            query,
            rec_L1,
            rec_L2,
            model_L1,
            model_L2
        )

        result = evaluate_recommendations(
            judge_model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            query=query,
            query_id=query_id,
            model_name=judge_info['model_name'],
            max_retries=JudgeConfig.MAX_RETRIES,
            retry_delay=JudgeConfig.RETRY_DELAY
        )

        combined_result = {
            'query_id': query_id,
            'query': query,
            'judge_model': judge_info['model_name'],
            'experiment': experiment_name,
            'rec_llm_L1': model_L1,
            'rec_llm_L2': model_L2,
            'evaluation': result.get('evaluation'),
            'success': result.get('success'),
            'token_stats': result.get('token_stats')
        }

        # Find and replace existing entry if it exists (e.g., a failed evaluation)
        # Otherwise append as a new entry
        existing_idx = next((i for i, e in enumerate(evaluations) if e.get('query_id') == query_id), None)
        if existing_idx is not None:
            evaluations[existing_idx] = combined_result
        else:
            evaluations.append(combined_result)

        save_evaluations(evaluations, output_file)

    # summary
    successful = sum(1 for e in evaluations if e.get('success', False))
    total_tokens = sum(e.get('token_stats', {}).get('total_tokens', 0) for e in evaluations)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total queries evaluated: {len(evaluations)}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {len(evaluations) - successful}")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Output file: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RecLLM recommendations using LLM-as-Judge"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to run. If not specified, lists available experiments."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments sequentially"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate (useful for testing)"
    )
    parser.add_argument(
        "-rerun_failed", "--rerun_failed", "--rerun-failed",
        action="store_true",
        help=(
            "Only re-evaluate queries that previously failed (requires existing output file). "
            "Accepts -rerun_failed (legacy), --rerun_failed, or --rerun-failed."
        )
    )

    args = parser.parse_args()

    # List experiments if none specified
    if not args.experiment and not args.all:
        print("Available experiments:")
        for exp_name in ExperimentConfig.list_experiments():
            exp_config = ExperimentConfig.get_experiment(exp_name)
            print(f"\n  {exp_name}:")
            print(f"    Judge: {exp_config['judge_model']}")
            print(f"    L1: {exp_config['rec_file_L1']}")
            print(f"    L2: {exp_config['rec_file_L2']}")
        print("\nUsage:")
        print("  python generate.py --experiment <experiment_name> [--max-queries N]")
        print("  python generate.py --all")
        return

    # Run experiments
    if args.all:
        experiments = ExperimentConfig.list_experiments()
        print(f"Running all {len(experiments)} experiments...\n")
        for exp_name in experiments:
            run_experiment(exp_name, max_queries=args.max_queries, rerun_failed=args.rerun_failed)
            print("\n")
    else:
        run_experiment(args.experiment, max_queries=args.max_queries, rerun_failed=args.rerun_failed)


if __name__ == "__main__":
    main()
