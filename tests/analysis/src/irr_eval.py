"""
Inter-Rater Reliability Evaluation using Krippendorff's Alpha

This module computes inter-rater agreement using Krippendorff's alpha,
which properly handles missing annotators and missing values.
"""

import pandas as pd
import numpy as np
import krippendorff


def compute_krippendorff_alpha(csv_path: str, metric: str = None) -> dict:
    """
    Compute Krippendorff's alpha for inter-rater agreement.

    Krippendorff's alpha is a reliability coefficient that:
    - Handles missing data (NaN values)
    - Works with any number of raters
    - Supports different measurement levels (nominal, ordinal, interval, ratio)

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing annotations
    metric : str, optional
        Specific metric to compute alpha for ('relevance', 'popularity',
        'diversity', 'sustainability'). If None, computes for all metrics.

    Returns:
    --------
    dict
        Dictionary with Krippendorff's alpha values for each metric
        Format: {metric_name: alpha_value}

    Notes:
    ------
    - NaN values are properly handled and excluded from computation
    - Missing annotators for specific queries are automatically handled
    - Uses ordinal level of measurement (appropriate for Likert-scale ratings)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Remove rows where annotator is NaN
    df = df[df['annotator'].notna()].copy()

    # Define the metrics to evaluate
    metrics = ['relevance', 'popularity', 'diversity', 'sustainability']

    if metric is not None:
        if metric not in metrics:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {metrics}")
        metrics = [metric]

    results = {}

    for metric_name in metrics:
        # Create a reliability matrix: queries × annotators
        # Pivot the data so that each row is a query and each column is an annotator
        pivot_df = df.pivot_table(
            index='query_id',
            columns='annotator',
            values=metric_name,
            aggfunc='first'  # In case of duplicates, take the first value
        )

        # Convert to numpy array and transpose to get annotators × queries
        # Krippendorff expects format: (raters × items)
        reliability_data = pivot_df.T.values

        # Compute Krippendorff's alpha
        # Use 'ordinal' level since the ratings are on an ordinal scale
        # NaN values are automatically handled by the library
        try:
            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement='ordinal'
            )
            results[metric_name] = alpha
        except Exception as e:
            print(f"Error computing alpha for {metric_name}: {e}")
            results[metric_name] = None

    return results


def compute_krippendorff_alpha_detailed(csv_path: str) -> pd.DataFrame:
    """
    Compute Krippendorff's alpha with detailed statistics for each metric.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing annotations

    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed statistics including:
        - metric: metric name
        - alpha: Krippendorff's alpha value
        - n_raters: number of raters
        - n_items: number of items (queries)
        - missing_pct: percentage of missing values
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Remove rows where annotator is NaN
    df = df[df['annotator'].notna()].copy()

    # Define the metrics to evaluate
    metrics = ['relevance', 'popularity', 'diversity', 'sustainability']

    detailed_results = []

    for metric_name in metrics:
        # Create a reliability matrix: queries × annotators
        pivot_df = df.pivot_table(
            index='query_id',
            columns='annotator',
            values=metric_name,
            aggfunc='first'
        )

        # Convert to numpy array and transpose
        reliability_data = pivot_df.T.values

        # Calculate statistics
        n_raters = reliability_data.shape[0]
        n_items = reliability_data.shape[1]
        total_cells = n_raters * n_items
        missing_cells = np.isnan(reliability_data).sum()
        missing_pct = (missing_cells / total_cells) * 100

        # Compute Krippendorff's alpha
        try:
            alpha = krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement='ordinal'
            )
        except Exception as e:
            print(f"Error computing alpha for {metric_name}: {e}")
            alpha = None

        detailed_results.append({
            'metric': metric_name,
            'alpha': alpha,
            'n_raters': n_raters,
            'n_items': n_items,
            'missing_cells': missing_cells,
            'missing_pct': round(missing_pct, 2)
        })

    return pd.DataFrame(detailed_results)


def compute_pairwise_agreement(csv_path: str, metric: str = None) -> pd.DataFrame:
    """
    Compute pairwise Krippendorff's alpha between all pairs of annotators.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing annotations
    metric : str, optional
        Specific metric to compute pairwise agreement for. If None, uses 'relevance'.

    Returns:
    --------
    pd.DataFrame
        Matrix of pairwise alpha values between annotators
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Remove rows where annotator is NaN
    df = df[df['annotator'].notna()].copy()

    if metric is None:
        metric = 'relevance'

    # Create a reliability matrix: queries × annotators
    pivot_df = df.pivot_table(
        index='query_id',
        columns='annotator',
        values=metric,
        aggfunc='first'
    )

    annotators = pivot_df.columns.tolist()
    n_annotators = len(annotators)

    # Initialize pairwise agreement matrix
    pairwise_alpha = pd.DataFrame(
        np.nan,
        index=annotators,
        columns=annotators
    )

    # Compute pairwise alpha
    for i, annotator1 in enumerate(annotators):
        for j, annotator2 in enumerate(annotators):
            if i == j:
                pairwise_alpha.loc[annotator1, annotator2] = 1.0
            elif i < j:
                # Get data for this pair of annotators
                pair_data = pivot_df[[annotator1, annotator2]].T.values

                try:
                    alpha = krippendorff.alpha(
                        reliability_data=pair_data,
                        level_of_measurement='ordinal'
                    )
                    pairwise_alpha.loc[annotator1, annotator2] = alpha
                    pairwise_alpha.loc[annotator2, annotator1] = alpha
                except Exception as e:
                    print(f"Error computing alpha for {annotator1} vs {annotator2}: {e}")

    return pairwise_alpha


if __name__ == "__main__":
    # Example usage
    csv_path = "../../../data/conv-trs/ecir-2026/human-eval/survey_parsed.csv"

    print("=" * 80)
    print("Inter-Rater Reliability Analysis using Krippendorff's Alpha")
    print("=" * 80)

    # Compute alpha for all metrics
    print("\n1. Krippendorff's Alpha for each metric:")
    print("-" * 80)
    results = compute_krippendorff_alpha(csv_path)
    for metric, alpha in results.items():
        if alpha is not None:
            print(f"{metric:15s}: α = {alpha:.4f}")
        else:
            print(f"{metric:15s}: Error computing alpha")

    # Compute detailed statistics
    print("\n2. Detailed Statistics:")
    print("-" * 80)
    detailed_df = compute_krippendorff_alpha_detailed(csv_path)
    print(detailed_df.to_string(index=False))

    # Interpretation guide
    print("\n3. Interpretation of Krippendorff's Alpha:")
    print("-" * 80)
    print("α ≥ 0.800: Reliable and can be used for drawing conclusions")
    print("0.667 ≤ α < 0.800: Tentative conclusions can be drawn")
    print("α < 0.667: Unreliable, should be discarded")
    print("α < 0: Systematic disagreement")

    # Compute pairwise agreement for relevance
    print("\n4. Pairwise Agreement (Relevance):")
    print("-" * 80)
    pairwise_df = compute_pairwise_agreement(csv_path, metric='relevance')
    print(pairwise_df.round(3).to_string())

