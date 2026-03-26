import pandas as pd
import numpy as np

def compute_mapping(df, dimensions=None, include_confidence=False, threshold=0.7):
    """
    Return a copy of df with *_score values mapped to 'L1'/'L2'/NaN.
    If include_confidence is True and a `{dim}_confidence` column exists,
    rows with confidence < threshold will have the corresponding `{dim}_score`
    treated as NaN before mapping.
    """
    if dimensions is None:
        dimensions = ["relevance", "sustainability", "popularity_balance", "diversity"]

    df = df.copy()
    mapping = {1: "L1", 2: "L1", -1: "L2", -2: "L2", 0: np.nan, -3: np.nan}

    for dim in dimensions:
        score_col = f"{dim}_score"
        conf_col = f"{dim}_confidence"

        if score_col not in df.columns:
            continue

        # If confidence filtering requested and confidence column exists,
        # mask low-confidence rows by setting the score to NaN
        if include_confidence and conf_col in df.columns:
            low_conf_mask = df[conf_col] < threshold
            if low_conf_mask.any():
                df.loc[low_conf_mask, score_col] = np.nan

        # Map numeric codes to labels/NaN safely on the copy
        df[score_col] = df[score_col].map(mapping).astype(object)

    return df

def count_best_matches(df, dimensions, best_col="best_list", include_confidence=False, threshold=0.7):
    """
    Count per-dimension how many rows have `{dim}_score` == best_col.
    If include_confidence is True, apply the confidence threshold before counting.
    Returns a DataFrame with columns: dimension, score_col, matches, n_valid, pct.
    """
    df_mapped = compute_mapping(df, dimensions=dimensions, include_confidence=include_confidence, threshold=threshold)
    results = []
    for dim in dimensions:
        score_col = f"{dim}_score"
        if score_col not in df_mapped.columns:
            results.append({"dimension": dim, "score_col": score_col, "matches": 0, "n_valid": 0, "pct": np.nan, "note": "missing"})
            continue

        valid_mask = df_mapped[score_col].notna() & df_mapped[best_col].notna()
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            matches = 0
            pct = np.nan
        else:
            matches = int((df_mapped.loc[valid_mask, score_col] == df_mapped.loc[valid_mask, best_col]).sum())
            pct = round(matches / n_valid * 100.0, 2)
        results.append({"dimension": dim, "score_col": score_col, "matches": matches, "n_valid": n_valid, "pct": pct})
    return pd.DataFrame(results)
