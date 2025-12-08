# experiments/run_task_a_title.py

import os
os.chdir("/content/MWAHAHA_Competition")
print("Working directory:", os.getcwd())

import zipfile
from dataclasses import dataclass
from typing import List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    adapter_repo_id: str = "An-di/qwen2_5_3b_jokes_lora"

    input_path: str = "../data/task-a-title.csv"
    output_dir: str = "outputs"
    output_filename: str = "task-a-title_predictions.csv"

    # generation settings
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    # where to copy the final zip on Google Drive
    drive_output_dir: str = "/content/drive/MyDrive/MWAHAHA_outputs"


cfg = InferenceConfig()

SYSTEM_PROMPT = (
    "You are a multilingual stand-up comedian. "
    "You write short, original jokes in English"
    "You ALWAYS obey the user’s constraints exactly (word inclusion, topic, language). "
    "You prefer concise setups and strong punchlines."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print("Loading tokenizer from:", cfg.adapter_repo_id)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.adapter_repo_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        print("Using CUDA (float16)")
    else:
        dtype = torch.float32
        device_map = "auto"
        print("CUDA not available, using CPU (float32)")

    print("Loading base model:", cfg.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print("Attaching LoRA adapter from:", cfg.adapter_repo_id)
    model = PeftModel.from_pretrained(base_model, cfg.adapter_repo_id)
    model.eval()

    print("Model device:", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
) -> List[str]:
    outputs: List[str] = []
    total = len(prompts)
    for i, p in enumerate(prompts):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Generating for example {i+1}/{total}")

        text = build_chat_prompt(tokenizer, p)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        # cut off prompt tokens
        new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(resp)

    return outputs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_and_zip(df_out: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    print("Saving predictions to:", out_path)
    # write as tab-separated, like the input
    df_out.to_csv(out_path, sep="\t", index=False)

    zip_path = out_path + ".zip"
    print("Zipping to:", zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=os.path.basename(out_path))

    return zip_path


def copy_to_drive(zip_path: str, drive_dir: str):
    try:
        from google.colab import drive as colab_drive  # type: ignore
    except ImportError:
        print("Not running in Colab, skipping Google Drive upload.")
        return

    if not os.path.exists("/content/drive"):
        print("Mounting Google Drive...")
        colab_drive.mount("/content/drive")

    os.makedirs(drive_dir, exist_ok=True)
    dest_path = os.path.join(drive_dir, os.path.basename(zip_path))

    import shutil
    shutil.copy2(zip_path, dest_path)
    print("Copied zip to Google Drive:", dest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data from:", cfg.input_path)
    # Your file is actually tab-separated
    df = pd.read_csv(cfg.input_path, sep="\t")

    # Use the 'headline' column as the prompt; keep all columns
    if "headline" not in df.columns:
        raise ValueError(f"'headline' column not found in {cfg.input_path}")
    prompts = df["headline"].astype(str).tolist()

    model, tokenizer = load_model_and_tokenizer()
    predictions = generate_responses(model, tokenizer, prompts)

    # keep original table, just append predictions
    df_out = df.copy()
    df_out["prediction"] = predictions

    zip_path = save_and_zip(df_out, cfg.output_dir, cfg.output_filename)
    copy_to_drive(zip_path, cfg.drive_output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
