from pathlib import Path

RUNNER_CONFIG = {
    "task": "title",
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "input_path": Path("data/task-a-title.csv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-a-title_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",
    "batch_size": 16,
    "noun_seed_base": 42,
    "replan_every": 5,
    "plan_decode": {
        "max_new_tokens": 80,
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
    "max_retries": 20,
    "retry_steps": [
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.10, "top_p": 0.99, "max_new_tokens": 56},

        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.10, "top_p": 0.99, "max_new_tokens": 56},

        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.10, "top_p": 0.99, "max_new_tokens": 56},

        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.10, "top_p": 0.99, "max_new_tokens": 56},

        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.10, "top_p": 0.99, "max_new_tokens": 56},
    ],
}
