from tests.analysis.src.utils import load_listwise_evaluations_df
from pathlib import Path

def extract_data(file_name, model_name):
    # Use absolute path from the script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent  # Go up from tests/analysis/src to project root
    file_path = project_root / "data" / "conv-trs" / "ecir-2026" / "direct-reasoner" / file_name

    print(f"📂 Loading from: {file_path}")
    load_listwise_evaluations_df(file_path_or_list=str(file_path), model_name=model_name)


if __name__ == "__main__":
    extract_data("gemini_eval_v1.json", "gemini")
    extract_data("gpt5_eval_v1.json", "gpt5")
    extract_data("deepseek_eval_v1.json", "deepseek")
    extract_data("calibration_deepseek_v1.json", "calibrated_deepseek")
    extract_data("calibration_gemini_v1.json", "calibrated_gemini")
    extract_data("calibration_gpt5_v1.json", "calibrated_gpt5")