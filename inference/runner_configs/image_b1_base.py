from pathlib import Path

RUNNER_CONFIG = {
    "task": "image_caption_b1",
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "lora_adapter_path": None,

    # Input is a tab separated values file with at least: id, url (or gif_url / gif_path)
    #"input_path": Path("data/task-b1.tsv"),
    "input_path": Path("data/test_subsets/task-b1.test100.csv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-b1_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",

    # Where the scene extractor writes: id, scene, noun1, noun2
    "scenes_output_csv": Path("data/task-b1.scenes.csv"),

    # Vision language extractor is called as a shell command:
    # python -m inference.utils.vl_scene_extractor --input_tsv ... --output_csv ...
    "vl_scene_extractor_module": "inference.utils.vl_scene_extractor",

    "vl_force_rerun": True,

    "batch_size": 16,
    "caption_max_words": 20,

    "plan_decode": {
        "max_new_tokens": 120,
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
        # Phase 1 (tries 1-3): concise + safe (helps 20-word limit)
        {"temperature": 0.80, "top_p": 0.92, "max_new_tokens": 28},
        {"temperature": 0.85, "top_p": 0.94, "max_new_tokens": 28},
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 32},

        # Phase 2 (tries 4-6): slightly more creative, still short
        {"temperature": 0.95, "top_p": 0.97, "max_new_tokens": 32},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 36},
        {"temperature": 1.05, "top_p": 0.99, "max_new_tokens": 36},
    ],

    "humor_policy": {
        "default_image": "irony",

        # keep your conditional pun logic via markers,
        # but if pun was chosen and fails once, go back to irony
        "image_pun_fallback_after": 1,

        "image_text_markers": ["sign", "label", "caption", "text", "words", "logo", "shirt", "screen", "poster", "menu"],
        "image_written_markers": ["says", "written", "printed", "reads"],
    },

}
