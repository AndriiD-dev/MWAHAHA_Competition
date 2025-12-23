from pathlib import Path

RUNNER_CONFIG = {
    "task": "caption_mm",

    # Your *fine-tuned* text model (or base model) that generates captions.
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "lora_adapter_path": "/content/drive/MyDrive/qwen_lora_jokes",  # or "Qwen/Qwen2.5-3B-Instruct"

    "input_path": Path("data/task-b1.tsv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-b-captions_predictions.tsv",

    "batch_size": 8,

    # caption generation decode (you can tune)
    "final_decode": {
        "max_new_tokens": 32,
        "min_new_tokens": 6,
        "temperature": 0.9,
        "top_p": 0.95,
    },

    # how many candidates per GIF to sample and rerank
    "candidates_per_row": 4,

    # internal validator limit (not shown to model)
    "caption_max_words": 20,

    # GIF frame extraction settings
    "gif_frames": {
        "max_frames": 4,
        "max_side": 512,
        "timeout_seconds": 60.0,
    },

    "max_retries": 3,
    "retry_steps": [
        {"temperature": 0.95, "top_p": 0.98, "max_new_tokens": 40},
        {"temperature": 1.00, "top_p": 0.99, "max_new_tokens": 48},
        {"temperature": 1.05, "top_p": 0.99, "max_new_tokens": 56},
    ],

    # optional (Colab Drive copy)
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",
}
