import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# 1️⃣ Mapping and constants
# ==========================================================

# example mapping (adjust values to your preferred scale)
COMPARISON_MAP = {
    "much more l1 than l2": 2,
    "much more l1": 2,
    "much more l2 than l1": -2,
    "much more l2": -2,
    "slightly more l1": 1,
    "slightly more l1 than l2": 1,
    "slightly more l2 than l1": -1,
    "slightly more l2": -1,
    "about the same": 0,
    "not sure / don't know": -3,
    "not sure / don’t know": -3,
    "not sure": -3,
    "don't know": -3
}


def _clean_col_name(s: str) -> str:
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")


def load_listwise_evaluations_df(file_path_or_list, model_name=None, version="v1"):
    """
    Load listwise evaluation JSON and flatten into a DataFrame (one row per query).
    Accepts either a path to a JSON file (list of entries) or the already-parsed list.
    """
    # load data if a file path string was passed
    if isinstance(file_path_or_list, str):
        with open(file_path_or_list, 'r') as f:
            data = json.load(f)
    else:
        data = file_path_or_list

    # keep only successful entries
    data = [item for item in data if item.get('success') == True]
    print(
        f"ℹ️ Loaded {len(data)} successful evaluation entries for {model_name or 'unknown model'}, version {version}.")

    rows = []
    for i, entry in enumerate(data):
        try:
            base = {
                "query_id": entry.get("query_id", f"unknown_{i}"),
                "query": entry.get("query", "").strip(),
                "judge_model": entry.get("judge_model", "unknown"),
                "experiment": entry.get("experiment", "N/A"),
                "rec_llm_L1": entry.get("rec_llm_L1", "unknown"),
                "rec_llm_L2": entry.get("rec_llm_L2", "unknown"),
                "best_list": entry.get("evaluation", {}).get("Best List", np.nan),
                "justification": entry.get("evaluation", {}).get("Justification", np.nan),
                "overall_confidence": entry.get("evaluation", {}).get("Overall Confidence", np.nan)
            }

            pairwise = entry.get("evaluation", {}).get("Pairwise_Comparisons", {})
            if not pairwise:
                print(f"⚠️ Entry {base['query_id']} missing 'Pairwise_Comparisons', skipping.")
                continue

            for dim_name, dim_vals in pairwise.items():
                clean = _clean_col_name(dim_name)
                comparison = dim_vals.get("Comparison", np.nan)
                explanation = dim_vals.get("Explanation", "")
                confidence = dim_vals.get("Confidence", np.nan)

                base[f"{clean}_comparison"] = comparison
                base[f"{clean}_explanation"] = explanation
                base[f"{clean}_confidence"] = confidence
                # base[f"{clean}_numeric_score"] = numeric_score
                # base[f"{clean}_is_unsure"] = is_unsure

            rows.append(base)

        except Exception as e:
            print(f"⚠️ Error parsing entry {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.drop(columns=["overall_comparison", "overall_explanation"], inplace=True)
    df.rename(columns={"justification": "overall_explanation"}, inplace=True)
    df = add_numeric_scores(df)

    # Create directory if it doesn't exist - use absolute path from file location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent  # Go up from tests/analysis/src to project root
    output_path = project_root / "data" / "conv-trs" / "ecir-2026" / "direct-reasoner" / "cleaned" / f"{model_name}_evals_cleaned_{version}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned evaluations to {output_path}")
    if df.empty:
        print("❌ No valid entries loaded! Check JSON structure.")
    return df


def add_numeric_scores(df):
    """Add numeric scores and is_unsure columns based on comparison mapping."""
    for col in df.columns:
        if col.endswith("_comparison"):
            numeric_col = col.replace("_comparison", "_score")

            df[numeric_col] = df[col].str.lower().map(COMPARISON_MAP)
    return df


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def set_paper_style():
    sns.set(style="white")
    # plt.subplots(figsize=(22, 12))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.labelsize'] = 38
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['xtick.labelsize'] = 38
    plt.rcParams['ytick.labelsize'] = 38
    plt.rcParams['legend.fontsize'] = 26
    plt.rcParams['figure.titlesize'] = 28
    plt.rcParams['lines.linewidth'] = 5.0
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
