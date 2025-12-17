from pathlib import Path

RUNNER_CONFIG = {
    "task": "two_words",
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "input_path": Path("data/task-a-two-words.csv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-a-two-words_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",
    "batch_size": 16,
    "plan_decode": {
        "max_new_tokens": 72,
        "min_new_tokens": 16,
        "temperature": 0.4,
        "top_p": 0.9,
    },
    "final_decode": {
        "max_new_tokens": 32,
        "min_new_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "max_retries": 5,
    "retry_steps": [
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
    ],
}
