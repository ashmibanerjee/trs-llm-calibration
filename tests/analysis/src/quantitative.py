import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import scikit_posthocs as sp
from statsmodels.stats.inter_rater import fleiss_kappa
DIMENSIONS =["relevance", "diversity", "popularity_balance", "sustainability"]

from scipy.stats import kruskal

def compute_kruskal_test(df_gpt, df_gemini, df_deepseek):
    results = []
    for dim in DIMENSIONS:
        scores_gpt = df_gpt[f'{dim}_score'].replace(-3, np.nan).dropna()
        scores_gemini = df_gemini[f'{dim}_score'].replace(-3, np.nan).dropna()
        scores_deepseek = df_deepseek[f'{dim}_score'].replace(-3, np.nan).dropna()

        stat, p = kruskal(scores_gpt, scores_gemini, scores_deepseek)
        significance = "Significant" if p < 0.01 else "Not significant"
        # combine into a single DataFrame
        data = pd.concat([
            pd.DataFrame({'model': 'GPT', 'score': scores_gpt}),
            pd.DataFrame({'model': 'Gemini', 'score': scores_gemini}),
            pd.DataFrame({'model': 'Deepseek', 'score': scores_deepseek})
        ], ignore_index=True)
        if significance == "Significant":
            dunn = sp.posthoc_dunn(data, val_col='score', group_col='model', p_adjust='bonferroni')
            print(f"\nDunn Post-hoc (Bonferroni corrected p-values): for dimension '{dim}'")
            print(dunn.round(4))

        results.append({
            'dimension': dim,
            'kruskal_H': stat,
            'p_value': p,
            'significance': significance,
        })
    significance_df = pd.DataFrame(results)
    print("\nKruskal-Wallis Test Results:")
    print(significance_df)


def compute_stats_scores(df, with_confidence=False):
    """
    Compute mean and variance per dimension (excluding unsure=-3 or NaN).
    Optionally compute weighted mean and weighted variance using confidence as weights.

    Returns:
        pd.DataFrame with columns: dimension, mean, variance, count
    """
    stats = []

    for dim in DIMENSIONS:
        score_col = f"{dim}_score"
        conf_col = f"{dim}_confidence"

        if score_col not in df.columns:
            print(f"⚠️ Missing column: {score_col}")
            continue

        # Filter out unsure / NaN
        valid_df = df.loc[(df[score_col].notna()) & (df[score_col] != -3)].copy()
        if valid_df.empty:
            stats.append({
                "dimension": dim,
                "mean": np.nan,
                "variance": np.nan,
                "count": 0
            })
            continue

        scores = valid_df[score_col]
        count = len(scores)

        if with_confidence and conf_col in df.columns:
            weights = valid_df[conf_col].fillna(0)
            weighted_mean = np.average(scores, weights=weights)
            weighted_var = np.average((scores - weighted_mean)**2, weights=weights)
            stats.append({
                "dimension": dim,
                "mean": weighted_mean,
                "variance": weighted_var,
                "count": count
            })
        else:
            mean_val = scores.mean(skipna=True)
            var_val = scores.var(ddof=1)
            stats.append({
                "dimension": dim,
                "mean": mean_val,
                "variance": var_val,
                "count": count
            })

    return pd.DataFrame(stats)

def compute_agreement(df1, df2, df3):
    """
    Robust inter-model agreement analysis.

    Aligns the three dataframes on `query_id` (outer merge) and computes:
      - counts and percentages for three-way and pairwise agreements (only where raters are present)
      - Fleiss' kappa computed on queries where all three ratings are present

    Returns a dict of Fleiss' kappas per dimension (NaN when not enough complete rows).
    """
    kappas = {}
    print("\nAgreement Analysis:")

    # don't mutate the global DIMENSIONS
    dims = list(DIMENSIONS) + ["best_list"]

    for dim in dims:
        col = dim if dim == "best_list" else f"{dim}_comparison"

        # prepare per-df (query_id, col) frames; if missing, produce NaN column so merges keep query ids
        def pick_col(df, idx):
            if col in df.columns:
                return df[["query_id", col]].rename(columns={col: f"{col}_{idx}"})
            else:
                # If the df doesn't have the column, return its query ids and an all-NaN rating column
                return df[["query_id"]].assign(**{f"{col}_{idx}": np.nan})

        a = pick_col(df1, 1)
        b = pick_col(df2, 2)
        c = pick_col(df3, 3)

        # outer merge to keep all query ids seen in any df
        m = pd.merge(a, b, on="query_id", how="outer")
        m = pd.merge(m, c, on="query_id", how="outer")

        c1 = f"{col}_1"
        c2 = f"{col}_2"
        c3 = f"{col}_3"

        total_queries = len(m)

        # Counts where all three present
        mask_all_three_present = m[c1].notna() & m[c2].notna() & m[c3].notna()
        n_all_three = int(mask_all_three_present.sum())
        all_three_eq = 0
        if n_all_three > 0:
            all_three_eq = int(((m.loc[mask_all_three_present, c1] == m.loc[mask_all_three_present, c2]) &
                                (m.loc[mask_all_three_present, c1] == m.loc[mask_all_three_present, c3])).sum())

        # Pairwise: compute only over rows where both raters are present
        def pair_stats(ci, cj):
            present = m[ci].notna() & m[cj].notna()
            n_present = int(present.sum())
            if n_present == 0:
                return 0, np.nan
            eq = int((m.loc[present, ci] == m.loc[present, cj]).sum())
            pct = (eq / n_present) * 100.0
            return eq, pct

        gpt_gem_count, gpt_gem_pct = pair_stats(c1, c2)
        gpt_ds_count, gpt_ds_pct = pair_stats(c1, c3)
        gem_ds_count, gem_ds_pct = pair_stats(c2, c3)

        none_equal_count = int(((~( (m[c1] == m[c2]) & (m[c1] == m[c3]) )) &
                               ~( (m[c1] == m[c2]) & (m[c1] != m[c3]) ) &
                               ~( (m[c1] == m[c3]) & (m[c1] != m[c2]) ) &
                               ~( (m[c2] == m[c3]) & (m[c1] != m[c2]) ) ).sum())

        print(f"\n=== {dim.upper()} ===")
        print(f"Total query ids aligned: {total_queries}; rows with all three present: {n_all_three}")
        if n_all_three > 0:
            print(f"All three agree (GPT=Gem=DS) [of rows with all three]: {all_three_eq} / {n_all_three} ({(all_three_eq/n_all_three)*100:.2f}%)")
        else:
            print("All three agree: 0 (no complete rows)")

        print(f"GPT=Gem (only where both present): {gpt_gem_count} ({gpt_gem_pct if not np.isnan(gpt_gem_pct) else 'N/A'}%)")
        print(f"GPT=DS  (only where both present): {gpt_ds_count} ({gpt_ds_pct if not np.isnan(gpt_ds_pct) else 'N/A'}%)")
        print(f"Gem=DS  (only where both present): {gem_ds_count} ({gem_ds_pct if not np.isnan(gem_ds_pct) else 'N/A'}%)")
        print(f"No pairwise equality detected (raw count over aligned rows): {none_equal_count} / {total_queries}")

        # Fleiss' kappa — only on rows with all three present
        if n_all_three == 0:
            print("Not enough complete rows for Fleiss' kappa.")
            kappas[dim] = np.nan
            continue

        rows_all = m.loc[mask_all_three_present, [c1, c2, c3]].copy()
        # determine categories
        cats = sorted(rows_all.stack().dropna().unique())
        cat_index = {c: i for i, c in enumerate(cats)}

        counts = np.zeros((len(rows_all), len(cats)), dtype=int)
        for i, row in enumerate(rows_all.itertuples(index=False)):
            vals = [getattr(row, c1), getattr(row, c2), getattr(row, c3)]
            for v in vals:
                counts[i, cat_index[v]] += 1

        try:
            k = fleiss_kappa(counts)
        except Exception as e:
            print(f"Error computing Fleiss' kappa for dimension {dim}: {e}")
            k = np.nan

        kappas[dim] = k
        print(f"Fleiss' Kappa for dimension '{dim}': {k if not np.isnan(k) else 'NaN'}: {interpret_kappa(k) if not np.isnan(k) else ''}")

    return kappas

def interpret_kappa(kappa: float) -> str:
    if kappa < 0.0:
        return "Poor agreement"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def pairwise_correlation(df1, df2, label1="Model1", label2="Model2"):
    """
    Compute Pearson correlation for each dimension between two DataFrames.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        label1: Label for first model
        label2: Label for second model

    Returns:
        pd.DataFrame with columns: dimension, pearson_r, p_value
    """
    results = []

    for dim in DIMENSIONS:
        col = f"{dim}_score"

        if col not in df1.columns or col not in df2.columns:
            print(f"⚠️ Missing column: {col} in one of the DataFrames")
            continue

        merged = pd.merge(
            df1[['query_id', col]],
            df2[['query_id', col]],
            on='query_id',
            suffixes=(f'_{label1}', f'_{label2}')
        )

        # Filter out unsure / NaN
        valid = merged[
            (merged[f'{col}_{label1}'].notna()) & (merged[f'{col}_{label1}'] != -3) &
            (merged[f'{col}_{label2}'].notna()) & (merged[f'{col}_{label2}'] != -3)
        ]

        if valid.empty:
            results.append({
                "dimension": dim,
                "pearson_r": np.nan,
                "p_value": np.nan
            })
            continue

        r, p = pearsonr(valid[f'{col}_{label1}'], valid[f'{col}_{label2}'])
        if p < 0.01:
            significance = "Significant"
        else:
            significance = "Not significant"
        results.append({
            "dimension": dim,
            "pearson_r": r,
            "p_value": p,
            "significance": significance
        })

    return pd.DataFrame(results)