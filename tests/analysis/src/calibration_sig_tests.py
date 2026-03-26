"""
Compute significance tests for score * confidence products before and after calibration.
For each dimension (relevance, sustainability, diversity, popularity_balance),
we test if the weighted scores (score * confidence) changed significantly after calibration.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


DIMENSIONS = ["relevance", "sustainability", "diversity", "popularity_balance"]


def load_data(model_name, data_dir):
    """Load before and after calibration data for a given model."""
    before_file = data_dir / f"{model_name}_evals_cleaned_v1.csv"
    after_file = data_dir / f"calibrated_{model_name}_evals_cleaned_v1.csv"

    before_df = pd.read_csv(before_file)
    after_df = pd.read_csv(after_file)

    return before_df, after_df


def compute_weighted_scores(df):
    """Compute score * confidence for each dimension."""
    weighted_df = df.copy()

    for dim in DIMENSIONS:
        score_col = f"{dim}_score"
        conf_col = f"{dim}_confidence"
        weighted_col = f"{dim}_weighted"

        if score_col in df.columns and conf_col in df.columns:
            # Handle NaN values - they will remain NaN in the product
            weighted_df[weighted_col] = df[score_col] * df[conf_col]
        else:
            print(f"Warning: Missing columns for {dim}")
            weighted_df[weighted_col] = np.nan

    return weighted_df


def paired_significance_test(before_scores, after_scores, dimension):
    """
    Perform paired significance tests (paired t-test and Wilcoxon signed-rank test).

    Args:
        before_scores: Array of weighted scores before calibration
        after_scores: Array of weighted scores after calibration
        dimension: Name of the dimension being tested

    Returns:
        Dictionary with test results
    """
    # Remove NaN values and ensure paired samples
    valid_mask = ~(np.isnan(before_scores) | np.isnan(after_scores))
    before_clean = before_scores[valid_mask]
    after_clean = after_scores[valid_mask]

    n = len(before_clean)

    if n < 2:
        return {
            "dimension": dimension,
            "n_pairs": n,
            "mean_before": np.nan,
            "mean_after": np.nan,
            "mean_diff": np.nan,
            "std_diff": np.nan,
            "t_statistic": np.nan,
            "t_pvalue": np.nan,
            "t_significant": "N/A",
            "wilcoxon_statistic": np.nan,
            "wilcoxon_pvalue": np.nan,
            "wilcoxon_significant": "N/A",
            "effect_size_cohens_d": np.nan
        }

    # Compute statistics
    mean_before = np.mean(before_clean)
    mean_after = np.mean(after_clean)
    diff = after_clean - before_clean
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(after_clean, before_clean)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        # Use zero_method='wilcox' to handle ties
        wilcoxon_result = stats.wilcoxon(after_clean, before_clean, zero_method='wilcox', alternative='two-sided')
        wilcoxon_stat = wilcoxon_result.statistic
        wilcoxon_pval = wilcoxon_result.pvalue
    except Exception as e:
        print(f"  Warning: Wilcoxon test failed for {dimension}: {e}")
        wilcoxon_stat = np.nan
        wilcoxon_pval = np.nan

    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff if std_diff != 0 else np.nan

    # Determine significance at α = 0.05
    t_sig = "Yes (p<0.05)" if t_pval < 0.05 else "No (p>=0.05)"
    wilcoxon_sig = "Yes (p<0.05)" if wilcoxon_pval < 0.05 else "No (p>=0.05)"

    return {
        "dimension": dimension,
        "n_pairs": n,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "t_significant": t_sig,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pval,
        "wilcoxon_significant": wilcoxon_sig,
        "effect_size_cohens_d": cohens_d
    }


def run_calibration_tests(model_name, data_dir):
    """Run significance tests for a single model."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}\n")

    # Load data
    before_df, after_df = load_data(model_name, data_dir)

    # Ensure both dataframes have the same query_ids and are aligned
    # Merge on query_id to ensure proper pairing
    merged = before_df.merge(
        after_df,
        on='query_id',
        suffixes=('_before', '_after')
    )

    print(f"Loaded {len(before_df)} before-calibration entries")
    print(f"Loaded {len(after_df)} after-calibration entries")
    print(f"Matched {len(merged)} paired entries\n")

    # Compute weighted scores for both
    before_weighted = compute_weighted_scores(merged[[col for col in merged.columns if col.endswith('_before')]])
    after_weighted = compute_weighted_scores(merged[[col for col in merged.columns if col.endswith('_after')]])

    # Rename columns for easier access
    for dim in DIMENSIONS:
        before_weighted[f"{dim}_weighted"] = merged[f"{dim}_score_before"] * merged[f"{dim}_confidence_before"]
        after_weighted[f"{dim}_weighted"] = merged[f"{dim}_score_after"] * merged[f"{dim}_confidence_after"]

    # Run tests for each dimension
    results = []
    for dim in DIMENSIONS:
        before_scores = before_weighted[f"{dim}_weighted"].values
        after_scores = after_weighted[f"{dim}_weighted"].values

        result = paired_significance_test(before_scores, after_scores, dim)
        results.append(result)

        # Print detailed results for this dimension
        print(f"Dimension: {dim.upper()}")
        print(f"  Sample size: {result['n_pairs']} paired observations")
        print(f"  Mean weighted score before: {result['mean_before']:.4f}")
        print(f"  Mean weighted score after:  {result['mean_after']:.4f}")
        print(f"  Mean difference (after - before): {result['mean_diff']:.4f}")
        print(f"  Std of differences: {result['std_diff']:.4f}")
        print(f"  Paired t-test: t={result['t_statistic']:.4f}, p={result['t_pvalue']:.6f} [{result['t_significant']}]")
        print(f"  Wilcoxon signed-rank: W={result['wilcoxon_statistic']:.1f}, p={result['wilcoxon_pvalue']:.6f} [{result['wilcoxon_significant']}]")
        print(f"  Effect size (Cohen's d): {result['effect_size_cohens_d']:.4f}")
        print()

    return pd.DataFrame(results)


def main():
    """Run significance tests for all models."""
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    data_dir = project_root / "data" / "conv-trs" / "ecir-2026" / "direct-reasoner" / "cleaned"
    output_dir = project_root / "data" / "artifacts" / "llm-eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["gpt5", "gemini", "deepseek"]

    all_results = {}

    for model in models:
        try:
            results_df = run_calibration_tests(model, data_dir)
            all_results[model] = results_df

            # Save individual model results
            output_file = output_dir / f"calibration_sig_tests_{model}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"✅ Saved results to {output_file}\n")

        except Exception as e:
            print(f"❌ Error processing {model}: {e}\n")
            continue

    # Create combined summary
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL MODELS")
    print("="*80 + "\n")

    for model, results_df in all_results.items():
        print(f"\n{model.upper()}:")
        summary = results_df[['dimension', 'mean_diff', 't_pvalue', 't_significant',
                              'wilcoxon_pvalue', 'wilcoxon_significant', 'effect_size_cohens_d']]
        print(summary.to_string(index=False))

    # Save combined results
    combined_df = pd.concat([df.assign(model=model) for model, df in all_results.items()], ignore_index=True)
    combined_output = output_dir / "calibration_sig_tests_all_models.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"\n✅ Saved combined results to {combined_output}")


if __name__ == "__main__":
    main()

