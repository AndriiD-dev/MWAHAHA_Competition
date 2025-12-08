# experiments/run_task_a_title.py

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

print("Project root:", PROJECT_ROOT)
print("Data dir    :", DATA_DIR)
print("Output dir  :", OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_repo_id: str = "An-di/qwen2_5_3b_jokes_lora"  # your HF LoRA repo

    input_path: Path = DATA_DIR / "task-a-title.csv"
    output_dir: Path = OUTPUT_DIR
    output_filename: str = "task-a-title_predictions.tsv"

    # generation settings
    max_new_tokens: int = 32     # shorter = faster
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    batch_size: int = 8          # adjust if you hit CUDA OOM

    # where to copy the final zip on Google Drive (Colab)
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
    print("Loading tokenizer from HF repo:", cfg.adapter_repo_id)
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

    print("Attaching LoRA adapter from HF repo:", cfg.adapter_repo_id)
    model = PeftModel.from_pretrained(base_model, cfg.adapter_repo_id)
    model.eval()

    print("Model main device:", next(model.parameters()).device)
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


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]


@torch.inference_mode()
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
) -> List[str]:
    outputs: List[str] = []
    total = len(prompts)
    start_all = time.time()

    for batch_index, batch_prompts in enumerate(batched(prompts, batch_size), start=1):
        batch_start = time.time()
        start_item = (batch_index - 1) * batch_size + 1
        end_item = min(start_item + len(batch_prompts) - 1, total)
        print(f"\n[Batch {batch_index}] items {start_item}-{end_item} / {total}")

        # build chat prompts
        texts = [build_chat_prompt(tokenizer, p) for p in batch_prompts]

        # tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # lengths of each prompt (for cutting off the prompt part later)
        input_lengths = attention_mask.sum(dim=1)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        # decode each item
        for i in range(len(batch_prompts)):
            new_tokens = gen_ids[i, input_lengths[i]:]
            resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(resp)

        batch_time = time.time() - batch_start
        elapsed = time.time() - start_all
        print(f"[Batch {batch_index}] done in {batch_time:.1f}s, "
              f"elapsed total {elapsed:.1f}s")

    print(f"\nAll batches done in {time.time() - start_all:.1f}s total.")
    return outputs


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_and_zip(df_out: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    print("Saving predictions to:", out_path)
    df_out.to_csv(out_path, sep="\t", index=False)

    zip_path = out_path.with_suffix(out_path.suffix + ".zip")
    print("Zipping to:", zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)

    return zip_path


def copy_to_drive(zip_path: Path, drive_dir: str):
    try:
        from google.colab import drive as colab_drive  # type: ignore
    except ImportError:
        print("Not running in Colab, skipping Google Drive upload.")
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("Mounting Google Drive...")
        colab_drive.mount("/content/drive")

    dest_dir = drive_root / Path(drive_dir).relative_to("/content/drive")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / zip_path.name

    import shutil
    shutil.copy2(zip_path, dest_path)
    print("Copied zip to Google Drive:", dest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data from:", cfg.input_path)
    df = pd.read_csv(cfg.input_path, sep="\t")

    if "headline" not in df.columns:
        raise ValueError(f"'headline' column not found in {cfg.input_path}")
    prompts = df["headline"].astype(str).tolist()
    print(f"Total prompts: {len(prompts)}")

    model, tokenizer = load_model_and_tokenizer()

    predictions = generate_responses(
        model,
        tokenizer,
        prompts,
        batch_size=cfg.batch_size,
    )

    df_out = df.copy()
    df_out["prediction"] = predictions

    zip_path = save_and_zip(df_out, cfg.output_dir, cfg.output_filename)
    copy_to_drive(zip_path, cfg.drive_output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
