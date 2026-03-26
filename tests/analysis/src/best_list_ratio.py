import pandas as pd
import numpy as np


def compute_ratio(df):
    """
    compute the ratio of the best list to the total aggregated scores (sum of confidence scores * scores)/4
    and then take the value counts of best list and the ratio of best list to total aggregated scores
    (categorize them into L1 if positive and L2 if negative) and compute the ratio of L1/L2
    :param df:
    :return:
    """
    df["weighted_aggregated_score"] = (df["relevance_confidence"] * df["relevance_score"] + \
                                       df["sustainability_confidence"] * df["sustainability_score"] + \
                                       df["diversity_confidence"] * df["diversity_score"] + \
                                       df["popularity_balance_confidence"] * df["popularity_balance_score"]) / 4
    df["favored_list_aggregated"] = np.where(df["weighted_aggregated_score"] >= 0, "L1", "L2")
    counts_aggregated = df["favored_list_aggregated"].value_counts()
    ratio_aggregated = counts_aggregated["L1"] / counts_aggregated["L2"]
    counts_best_list = df["best_list"].value_counts()
    ratio_best_list = counts_best_list["L1"] / counts_best_list["L2"]
    return ratio_best_list, ratio_aggregated


def get_data(file_path):
    # print(f"\tLoading data from {file_path}")
    return pd.read_csv(file_path)


def run(model_name):
    print(f"\t========== before calibration ==========")
    file_path = f"../../../data/conv-trs/ecir-2026/direct-reasoner/cleaned/{model_name}_evals_cleaned_v1.csv"
    df = get_data(file_path)
    ratio_best_list, ratio_aggregated = compute_ratio(df)
    print(f"\t\tRatio of L1/L2 using aggregated scores: {ratio_aggregated}")
    print(f"\t\tRatio of L1/L2 using best list: {ratio_best_list}")
    print(f"\t========== after calibration ==========")
    if model_name == "gemini2.5pro":
        model_name = "gemini"
    file_path = f"../../../data/conv-trs/ecir-2026/direct-reasoner/cleaned/calibrated_{model_name}_evals_cleaned_v1.csv"
    df = get_data(file_path)
    ratio_best_list, ratio_aggregated = compute_ratio(df)
    print(f"\t\tRatio of L1/L2 using aggregated scores: {ratio_aggregated}")
    print(f"\t\tRatio of L1/L2 using best list: {ratio_best_list}")


def main():
    model_names = ["gemini", "gpt5", "deepseek"]
    for model_name in model_names:
        print(f"\n{model_name}")
        run(model_name)

if __name__ == "__main__":
    main()
