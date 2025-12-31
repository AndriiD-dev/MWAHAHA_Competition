from pathlib import Path

RUNNER_CONFIG = {
    "task": "two_words",
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "lora_adapter_path": None,

    "input_path": Path("data/task-a-two-words.csv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-a-two-words_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",

    "batch_size": 16,

    "plan_decode": {
        "max_new_tokens": 180,
        "min_new_tokens": 64,
        "temperature": 0.4,
        "top_p": 0.9,
    },
    "final_decode": {
        "max_new_tokens": 32,
        "min_new_tokens": 6,
        "temperature": 0.8,
        "top_p": 0.95,
    },

    "max_retries": 6,
    "retry_steps": [
        # Phase 1 (tries 1-3): safer decoding for pun
        {"temperature": 0.80, "top_p": 0.92, "max_new_tokens": 36},
        {"temperature": 0.85, "top_p": 0.94, "max_new_tokens": 36},
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 40},

        # Phase 2 (tries 4-6): more freedom (useful after humor switch to irony)
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 44},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.99, "max_new_tokens": 52},
    ],

    "humor_policy": {
        "default_two_words": "pun",
        "two_words_switch_after": 3,  # pun for first half, then irony
    },

}
