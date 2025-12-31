from pathlib import Path

RUNNER_CONFIG = {
    "task": "headline",  # was "title"
    "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
    "lora_adapter_path": None,

    #"input_path": Path("data/task-a-title.csv"),
    "input_path": Path("data/test_subsets/task-a-title.test100.csv"),
    "output_dir": Path("outputs"),
    "output_filename": "task-a-title_predictions_base.tsv",
    "drive_output_dir": "MyDrive/MWAHAHA_Competition/outputs",

    "batch_size": 16,
    "noun_seed_base": 42,

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

    "max_retries": 10,
    "retry_steps": [
        # Phase 1 (tries 1-5): satire, controlled
        {"temperature": 0.80, "top_p": 0.92, "max_new_tokens": 40},
        {"temperature": 0.85, "top_p": 0.94, "max_new_tokens": 40},
        {"temperature": 0.90, "top_p": 0.95, "max_new_tokens": 44},
        {"temperature": 0.92, "top_p": 0.96, "max_new_tokens": 44},
        {"temperature": 0.95, "top_p": 0.97, "max_new_tokens": 48},

        # Phase 2 (tries 6-10): irony fallback, more freedom
        {"temperature": 0.98, "top_p": 0.98, "max_new_tokens": 48},
        {"temperature": 1.00, "top_p": 0.98, "max_new_tokens": 52},
        {"temperature": 1.02, "top_p": 0.99, "max_new_tokens": 52},
        {"temperature": 1.05, "top_p": 0.99, "max_new_tokens": 56},
        {"temperature": 1.08, "top_p": 0.99, "max_new_tokens": 56},
    ],

    "humor_policy": {
        "default_headline": "satire",
        "headline_switch_after": 5,  # satire first half, then irony
    },


    # You can tune these lists without touching runner code
    "headline_public_markers": [
        "government", "minister", "president", "parliament", "senate", "congress",
        "election", "campaign", "policy", "law", "court", "supreme", "police",
        "war", "military", "company", "corporate", "ceo", "platform",
        "artificial intelligence", "technology", "market", "economy", "study", "research",
    ],
    "headline_personal_markers": [
        "student", "teacher", "family", "parents", "customer",
        "restaurant", "flight", "hotel", "home", "school", "office",
    ],
}
