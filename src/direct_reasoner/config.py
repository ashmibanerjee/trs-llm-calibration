"""Configuration for direct reasoner experiments."""

from pathlib import Path


class ExperimentConfig:
    """Configuration for evaluation experiments."""

    # Define experiment configurations
    EXPERIMENTS = {
        # New experiments that use the selected queries JSON and run the two judges
        "gemini_pro_evaluations_v1": {
            "judge_model": "gemini-2.5-pro",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "run_common_human_queries_gemini.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "gpt5-evaluations_v1": {
            "judge_model": "gpt-5",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "run_common_human_queries_gpt5.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "deepseek_evaluations_v1": {
            "judge_model": "deepseek-v3",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "run_common_human_queries_deepseek.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "gemini_pro_evaluations_v2": {
            "judge_model": "gemini-2.5-pro",
            "rec_file_L1": "claude_4_sonnet_recommendations.json",
            "rec_file_L2": "qwen_3_next_80b_recommendations.json",
            "output_file": "gemini_eval_v2.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "gpt5-evaluations_v2": {
            "judge_model": "gpt-5",
            "rec_file_L1": "claude_4_sonnet_recommendations.json",
            "rec_file_L2": "qwen_3_next_80b_recommendations.json",
            "output_file": "gpt5_eval_v2.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "deepseek_evaluations_v2": {
            "judge_model": "deepseek-v3",
            "rec_file_L1": "claude_4_sonnet_recommendations.json",
            "rec_file_L2": "qwen_3_next_80b_recommendations.json",
            "output_file": "deepseek_eval_v2.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-listwise"
        },
        "run_item-wise_gemini": {
            "judge_model": "gemini-2.5-pro",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "item-wise_gemini.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-itemwise"
        },
        "run_item-wise_deepseek": {
            "judge_model": "deepseek-v3",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "item-wise_deepseek.json",
            "use_selected_queries": True,
            "prompts_dir": "direct-reasoner-itemwise"
        },
        "run_calibration_gemini": {
            "judge_model": "gemini-2.5-pro",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "calibration_gemini_v1.json",
            "use_selected_queries": True,
            "prompts_dir": "calibration"
        },
        "run_calibration_gpt": {
            "judge_model": "gpt-5",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "calibration_gpt5_v1.json",
            "use_selected_queries": True,
            "prompts_dir": "calibration"
        },
        "run_calibration_deepseek": {
            "judge_model": "deepseek-v3",
            "rec_file_L1": "gemini_2_5_flash_recommendations.json",
            "rec_file_L2": "gpt_4o_recommendations.json",
            "output_file": "calibration_deepseek_v1.json",
            "use_selected_queries": True,
            "prompts_dir": "calibration"
        },
        "run_calibration_deepseek_v2": {
            "judge_model": "deepseek-v3",
            "rec_file_L1": "claude_4_sonnet_recommendations.json",
            "rec_file_L2": "qwen_3_next_80b_recommendations.json",
            "output_file": "calibration_deepseek_v2.json",
            "use_selected_queries": True,
            "prompts_dir": "calibration"
        },
    }

    @classmethod
    def get_experiment(cls, experiment_name: str) -> dict:
        """Get experiment configuration by name."""
        if experiment_name not in cls.EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Available: {list(cls.EXPERIMENTS.keys())}"
            )
        return cls.EXPERIMENTS[experiment_name]

    @classmethod
    def list_experiments(cls) -> list:
        """List all available experiments."""
        return list(cls.EXPERIMENTS.keys())


class PathConfig:
    """Path configuration for direct reasoner."""

    BASE_DIR = Path(__file__).parent.parent.parent
    REC_LLM_DIR = BASE_DIR / "data" / "conv-trs" / "ecir-2026" / "rec-llm"
    OUTPUT_DIR = BASE_DIR / "data" / "conv-trs" / "ecir-2026" / "direct-reasoner"
    PROMPTS_DIR = BASE_DIR / "prompts" / "direct-reasoner-idk"
    # Path to JSON file containing selected queries to restrict evaluation to
    SELECTED_QUERIES_FILE = BASE_DIR / "data" / "conv-trs" / "ecir-2026" / "selected_queries" / "filtered_queries.json"

    @classmethod
    def get_rec_file(cls, filename: str) -> Path:
        """Get path to recommendation file."""
        return cls.REC_LLM_DIR / filename

    @classmethod
    def get_output_file(cls, filename: str) -> Path:
        """Get path to output file."""
        return cls.OUTPUT_DIR / filename


class JudgeConfig:
    """Configuration for judge model parameters."""

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    TEMPERATURE = 0.0  # Deterministic evaluation
