import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, ttest_rel, kruskal, spearmanr
from sklearn.metrics import cohen_kappa_score


# Constants
ANNOTATORS = ["Ewald", "Wolfgang"]
DIMENSIONS = ["diversity", "relevance", "sustainability", "popularity_mix"]
SCORE_RANGE = [-2, -1, 0, 1, 2]


def _get_valid_scores(df, annotator, dimension, exclude_minus_3=True):
    """Helper function to extract valid scores for an annotator and dimension."""
    col_name = f"{annotator}_{dimension}_score"
    if col_name not in df.columns:
        return pd.Series(dtype=float)

    scores = df[col_name]
    if exclude_minus_3:
        scores = scores[(scores != -3) & scores.notna()]
    else:
        scores = scores[scores.notna()]

    return scores


def _get_paired_scores(df, dimension, exclude_minus_3=True):
    """Helper function to extract paired scores for both annotators."""
    col_annotator1 = f"{ANNOTATORS[0]}_{dimension}_score"
    col_annotator2 = f"{ANNOTATORS[1]}_{dimension}_score"

    if col_annotator1 not in df.columns or col_annotator2 not in df.columns:
        return None, None, 0

    paired_df = df[[col_annotator1, col_annotator2]].copy()

    # Build validity mask
    valid_mask = paired_df[col_annotator1].notna() & paired_df[col_annotator2].notna()
    if exclude_minus_3:
        valid_mask = valid_mask & (paired_df[col_annotator1] != -3) & (paired_df[col_annotator2] != -3)

    paired_valid = paired_df[valid_mask]

    if len(paired_valid) == 0:
        return None, None, 0

    return paired_valid[col_annotator1].values, paired_valid[col_annotator2].values, len(paired_valid)


def compute_basic_stats(df):
    """
    Compute basic descriptive statistics per dimension for both annotators.

    Args:
        df: DataFrame with columns like {annotator}_{dimension}_score

    Returns:
        DataFrame with mean, std, count per dimension (combined across annotators)
    """
    mean_scores = {}

    for dim in DIMENSIONS:
        all_scores = []

        for annotator in ANNOTATORS:
            valid_scores = _get_valid_scores(df, annotator, dim)
            all_scores.extend(valid_scores.tolist())

        # Compute mean across both annotators
        if all_scores:
            mean_scores[dim] = {
                'mean': np.mean(all_scores),
                'count': len(all_scores),
                'std': np.std(all_scores, ddof=1)
            }
        else:
            mean_scores[dim] = {
                'mean': np.nan,
                'count': 0,
                'std': np.nan
            }

    # Convert to DataFrame for better display
    mean_scores_df = pd.DataFrame(mean_scores).T
    mean_scores_df.index.name = 'dimension'
    mean_scores_df = mean_scores_df.reset_index()

    print("\n" + "="*80)
    print("BASIC STATISTICS (BOTH ANNOTATORS COMBINED)")
    print("="*80)
    print("\nMean Scores per Dimension (excluding -3 = 'not sure'):")
    print(mean_scores_df.to_string(index=False))

    return mean_scores_df


def compute_agreement(df):
    """
    Compute inter-annotator agreement metrics per dimension.

    Includes:
    - Exact agreement count and percentage
    - Within-1 agreement (scores differ by at most 1)
    - Cohen's Kappa with interpretation
    - Score-by-score agreement breakdown

    Args:
        df: DataFrame with columns like {annotator}_{dimension}_score

    Returns:
        DataFrame with agreement statistics per dimension
    """
    agreement_results = []

    print("\n" + "=" * 80)
    print("INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("=" * 80)

    for dim in DIMENSIONS:
        scores1, scores2, n_pairs = _get_paired_scores(df, dim)

        if scores1 is None:
            print(f"\n⚠️ Missing columns for dimension: {dim}")
            continue

        if n_pairs == 0:
            print(f"\n⚠️ No valid paired ratings for dimension: {dim}")
            continue

        # Exact agreement count
        exact_agreement = (scores1 == scores2).sum()
        percent_agreement = (exact_agreement / n_pairs) * 100

        # Cohen's Kappa
        kappa = cohen_kappa_score(scores1, scores2)

        # Kappa interpretation
        if kappa < 0:
            kappa_interp = "Poor (worse than random)"
        elif kappa < 0.20:
            kappa_interp = "Slight"
        elif kappa < 0.40:
            kappa_interp = "Fair"
        elif kappa < 0.60:
            kappa_interp = "Moderate"
        elif kappa < 0.80:
            kappa_interp = "Substantial"
        else:
            kappa_interp = "Almost perfect"

        # Within-1 agreement (scores differ by at most 1)
        within_1 = (np.abs(scores1 - scores2) <= 1).sum()
        percent_within_1 = (within_1 / n_pairs) * 100

        # Agreement by specific score value
        score_agreements = {}
        for score in SCORE_RANGE:
            # Count where both annotators gave this specific score
            both_gave_score = ((scores1 == score) & (scores2 == score)).sum()
            # Count where at least one annotator gave this score
            either_gave_score = ((scores1 == score) | (scores2 == score)).sum()

            if either_gave_score > 0:
                score_agreement_pct = (both_gave_score / either_gave_score) * 100
            else:
                score_agreement_pct = 0

            score_agreements[f'score_{score}_agreement'] = both_gave_score
            score_agreements[f'score_{score}_total'] = either_gave_score
            score_agreements[f'score_{score}_pct'] = score_agreement_pct

        agreement_results.append({
            'dimension': dim,
            'total_paired_ratings': n_pairs,
            'exact_agreement': exact_agreement,
            'exact_agreement_pct': percent_agreement,
            'within_1_agreement': within_1,
            'within_1_agreement_pct': percent_within_1,
            'cohens_kappa': kappa,
            'kappa_interpretation': kappa_interp,
            **score_agreements
        })

        print(f"\n{dim.upper()}:")
        print(f"  Total paired ratings: {n_pairs}")
        print(f"  Exact agreement: {exact_agreement} ({percent_agreement:.1f}%)")
        print(f"  Within-1 agreement: {within_1} ({percent_within_1:.1f}%)")
        print(f"  Cohen's Kappa: {kappa:.3f} ({kappa_interp})")
        print(f"\n  Agreement by score:")
        for score in SCORE_RANGE:
            both = score_agreements[f'score_{score}_agreement']
            total = score_agreements[f'score_{score}_total']
            pct = score_agreements[f'score_{score}_pct']
            print(f"    Score {score:+d}: {both}/{total} ({pct:.1f}%)")

    # Create summary DataFrame
    agreement_df = pd.DataFrame(agreement_results)

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    return agreement_df


def compute_significance(df):
    """
    Test if the two annotators' score distributions are significantly different per dimension.

    Performs Wilcoxon signed-rank test (paired samples) - MOST APPROPRIATE for paired ratings.

    Args:
        df: DataFrame with columns like {annotator}_{dimension}_score

    Returns:
        DataFrame with test statistics and p-values per dimension
    """
    print("\n" + "="*80)
    print("DISTRIBUTION DIFFERENCE ANALYSIS (SIGNIFICANCE TESTING)")
    print("="*80)

    results = []

    for dim in DIMENSIONS:
        # Get all valid scores (exclude -3 and NaN)
        scores1_all = _get_valid_scores(df, ANNOTATORS[0], dim)
        scores2_all = _get_valid_scores(df, ANNOTATORS[1], dim)

        # Get paired scores (both annotators rated the same query)
        scores1_paired, scores2_paired, n_pairs = _get_paired_scores(df, dim)

        if scores1_paired is None:
            print(f"\n⚠️ Missing columns for dimension: {dim}")
            continue

        # Descriptive statistics
        mean1 = scores1_all.mean()
        mean2 = scores2_all.mean()
        std1 = scores1_all.std(ddof=1)
        std2 = scores2_all.std(ddof=1)
        median1 = scores1_all.median()
        median2 = scores2_all.median()

        # Wilcoxon signed-rank test (paired samples - more appropriate for same queries)
        if n_pairs > 0:
            # Check if there are any differences (Wilcoxon requires at least some differences)
            if np.any(scores1_paired != scores2_paired):
                statistic_w, p_value_w = wilcoxon(scores1_paired, scores2_paired, alternative='two-sided')
                significant_w = "Yes" if p_value_w < 0.05 else "No"
            else:
                statistic_w, p_value_w, significant_w = np.nan, 1.0, "No"
        else:
            statistic_w, p_value_w, significant_w = np.nan, np.nan, "N/A"

        results.append({
            'dimension': dim,
            'ewald_mean': mean1,
            'ewald_std': std1,
            'ewald_median': median1,
            'ewald_n': len(scores1_all),
            'wolfgang_mean': mean2,
            'wolfgang_std': std2,
            'wolfgang_median': median2,
            'wolfgang_n': len(scores2_all),
            'mean_diff': mean1 - mean2,
            'median_diff': median1 - median2,
            'wilcoxon_statistic': statistic_w,
            'wilcoxon_pvalue': p_value_w,
            'wilcoxon_significant': significant_w,
            'paired_n': n_pairs
        })

        print(f"\n{dim.upper()}:")
        print(f"  Ewald:    mean={mean1:.2f}, std={std1:.2f}, median={median1:.1f}, n={len(scores1_all)}")
        print(f"  Wolfgang: mean={mean2:.2f}, std={std2:.2f}, median={median2:.1f}, n={len(scores2_all)}")
        print(f"  Mean difference: {mean1 - mean2:+.2f}")
        print(f"  Median difference: {median1 - median2:+.1f}")
        print(f"\n  Wilcoxon signed-rank test (paired, n={n_pairs}):")
        print(f"    Statistic: {statistic_w:.2f}, p-value: {p_value_w:.4f}")
        print(f"    Significantly different? {significant_w} (α=0.05)")

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print("\nInterpretation:")
    print("- Wilcoxon signed-rank: Tests if paired differences are symmetric around zero")
    print("- Wilcoxon is MOST APPROPRIATE since the same queries were rated by both annotators")
    print("- Significant if p < 0.05")

    return results_df


def compute_rank_correlation(df):
    """
    Compute Spearman's rank correlation coefficient between the two annotators per dimension.

    Spearman's rho measures the monotonic relationship between two ranked variables.
    It assesses how well the relationship between two variables can be described
    using a monotonic function.

    Args:
        df: DataFrame with columns like {annotator}_{dimension}_score

    Returns:
        DataFrame with Spearman's rho, p-values, and interpretation per dimension
    """
    results = []

    for dim in DIMENSIONS:
        # Get paired scores (both annotators rated the same query, exclude -3)
        scores1, scores2, n_pairs = _get_paired_scores(df, dim)

        if scores1 is None:
            print(f"\n⚠️ Missing columns for dimension: {dim}")
            continue

        if n_pairs < 2:
            print(f"\n⚠️ Insufficient paired ratings for dimension: {dim}")
            continue

        # Compute Spearman's rank correlation
        rho, p_value = spearmanr(scores1, scores2)

        # Interpretation of correlation strength
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            strength = "Negligible"
        elif abs_rho < 0.3:
            strength = "Weak"
        elif abs_rho < 0.5:
            strength = "Moderate"
        elif abs_rho < 0.7:
            strength = "Strong"
        else:
            strength = "Very strong"

        # Direction
        if rho > 0:
            direction = "positive"
        elif rho < 0:
            direction = "negative"
        else:
            direction = "none"

        interpretation = f"{strength} {direction} correlation"

        # Statistical significance
        significant = "Yes" if p_value < 0.05 else "No"

        results.append({
            'dimension': dim,
            'spearman_rho': rho,
            'p_value': p_value,
            'significant': significant,
            'correlation_strength': strength,
            'direction': direction,
            'interpretation': interpretation,
            'n_pairs': n_pairs
        })

        print(f"\n{dim.upper()}:")
        print(f"  Spearman's rho: {rho:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {significant} (α=0.05)")
        print(f"  Interpretation: {interpretation}")
        print(f"  Number of paired ratings: {n_pairs}")

    # Create summary DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print("\nInterpretation Guide:")
    print("- Spearman's rho ranges from -1 to +1")
    print("- +1: Perfect positive monotonic relationship")
    print("- 0: No monotonic relationship")
    print("- -1: Perfect negative monotonic relationship")
    print("\nStrength interpretation:")
    print("- |rho| < 0.1: Negligible")
    print("- 0.1 ≤ |rho| < 0.3: Weak")
    print("- 0.3 ≤ |rho| < 0.5: Moderate")
    print("- 0.5 ≤ |rho| < 0.7: Strong")
    print("- |rho| ≥ 0.7: Very strong")

    return results_df
