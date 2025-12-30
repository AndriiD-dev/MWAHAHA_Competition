from pathlib import Path

RUNNER_CONFIG = {
    "task": "image_caption_b2",
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "lora_adapter_path": None,

    # Input is a tab separated values file with at least: id, url (or gif_url / gif_path), prompt (or prompt_text)
    "input_path": Path("data/task-b2.tsv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-b2_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",

    "scenes_output_csv": Path("data/task-b2.scenes.csv"),

    "vl_scene_extractor_module": "inference.utils.vl_scene_extractor",
    "vl_force_rerun": False,

    "batch_size": 16,
    "caption_max_words": 20,

    "plan_decode": {
        "max_new_tokens": 80,
        "min_new_tokens": 16,
        "temperature": 0.4,
        "top_p": 0.9,
    },
    "final_decode": {
        "max_new_tokens": 40,
        "min_new_tokens": 6,
        "temperature": 0.9,
        "top_p": 0.95,
    },

    "max_retries": 6,
    "retry_steps": [
        # Phase 1: safe + short
        {"temperature": 0.80, "top_p": 0.92, "max_new_tokens": 30},
        {"temperature": 0.85, "top_p": 0.94, "max_new_tokens": 30},
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 34},

        # Phase 2: more creative, still bounded
        {"temperature": 0.95, "top_p": 0.97, "max_new_tokens": 34},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 38},
        {"temperature": 1.05, "top_p": 0.99, "max_new_tokens": 38},
    ],

    "humor_policy": {
        "default_image": "irony",
        "image_pun_fallback_after": 1,
        "image_text_markers": ["sign", "label", "caption", "text", "words", "logo", "shirt", "screen", "poster", "menu"],
        "image_written_markers": ["says", "written", "printed", "reads"],
    },

}
