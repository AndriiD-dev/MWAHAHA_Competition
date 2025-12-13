from __future__ import annotations

import os
import re
import time
import zipfile
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


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
# Hugging Face authentication (for private repositories)
# ---------------------------------------------------------------------------

def get_hf_token() -> str | None:
    return (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )


def hf_auth_kwargs(for_callable) -> dict:
    """
    Transformers and Hugging Face Hub changed parameter names across versions.
    This filters keyword arguments to the ones supported by the callable.
    """
    tok = cfg.huggingface_token  # cfg is defined below
    if not tok:
        return {}

    raw = {
        "token": tok,            # newer
        "use_auth_token": tok,   # older
    }

    try:
        params = inspect.signature(for_callable).parameters
        return {k: v for k, v in raw.items() if k in params}
    except Exception:
        return {"token": tok}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    # Base model and adapter (low rank adaptation)
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    # Your adapter pushed to the Hugging Face Hub
    adapter_id: str = "An-di/qwen2_5_3b_jokes_lora"

    # Optional: if you have the adapter locally instead of the Hub
    # adapter_local_path: Path = Path("../models/qwen_lora_jokes")
    adapter_local_path: Path | None = None

    # Optional: merge adapter weights into the base model for faster inference.
    # This increases memory usage slightly and is not always compatible with some device mapping setups.
    merge_adapter: bool = False

    input_path: Path = DATA_DIR / "task-a-title.csv"
    output_dir: Path = OUTPUT_DIR
    output_filename: str = "task-a-title_predictions_lora.tsv"

    # Decoding (keep quality)
    max_new_tokens: int = 32
    min_new_tokens: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    # Throughput
    batch_size: int = 16  # increase if you have headroom; length bucketing makes this cheaper

    # “Zero empty cells” policy
    max_retries: int = 1  # reduce extra GPU passes
    retry_settings: Tuple[Tuple[float, float, int], ...] = (
        (1.0, 0.98, 48),
    )

    drive_output_dir: str = "/content/drive/MyDrive/MWAHAHA_outputs"

    # Token is required if your adapter repository is private
    huggingface_token: str | None = field(default_factory=get_hf_token)


cfg = InferenceConfig()

# Keep the exact prompt style from the clean Qwen inference script
SYSTEM_PROMPT = (
    "You are a stand-up comedian. Write ONE original joke in English for the user headline. "
    "Return exactly one line under 30 words. No preface, no explanation, no emojis. "
    "Do NOT include any role labels or speaker tags (for example: assistant, user, system). "
    "Do NOT start with 'assistant' or 'assistant:'. Output only the joke text. "
    "Follow user constraints. Avoid hate, slurs, explicit sex, and graphic violence."
)


# ---------------------------------------------------------------------------
# Speed knobs (no quality loss in practice)
# ---------------------------------------------------------------------------

def configure_fast_kernels():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Model loading (base + adapter)
# ---------------------------------------------------------------------------

def load_tokenizer():
    """
    Prefer tokenizer saved alongside the adapter (if present), otherwise use base model tokenizer.
    """
    sources: List[str] = []
    if cfg.adapter_local_path is not None:
        sources.append(str(cfg.adapter_local_path))
    sources.append(cfg.adapter_id)
    sources.append(cfg.base_model_id)

    last_err: Exception | None = None
    for src in sources:
        try:
            print("Loading tokenizer from:", src)
            tokenizer = AutoTokenizer.from_pretrained(
                src,
                trust_remote_code=True,
                **hf_auth_kwargs(AutoTokenizer.from_pretrained),
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load tokenizer from any source. Last error: {last_err}")


def load_model_and_tokenizer():
    tokenizer = load_tokenizer()

    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
        print("Using CUDA (float16)")
    else:
        torch_dtype = torch.float32
        device_map = "auto"
        print("CUDA not available, using CPU (float32)")

    # Prefer faster attention implementations if available
    attn_impl_candidates = ["flash_attention_2", "sdpa", "eager"]
    last_err = None
    base_model = None

    for attn_impl in attn_impl_candidates:
        try:
            print(f"Loading base model with attention implementation: {attn_impl}")
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                **hf_auth_kwargs(AutoModelForCausalLM.from_pretrained),
            )
            break
        except Exception as e:
            last_err = e
            base_model = None

    if base_model is None:
        raise RuntimeError(
            f"Failed to load base model with any attention implementation. Last error: {last_err}"
        )

    print("Attaching adapter...")

    adapter_source = cfg.adapter_local_path if cfg.adapter_local_path is not None else cfg.adapter_id
    model = PeftModel.from_pretrained(
        base_model,
        adapter_source,
        is_trainable=False,
        **hf_auth_kwargs(PeftModel.from_pretrained),
    )

    if cfg.merge_adapter:
        print("Merging adapter into base model (merge_adapter=True)")
        model = model.merge_and_unload()

    model.eval()

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    print("Model main device:", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt + batching
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # IMPORTANT: do not add another generation prompt (we already included assistant)
        add_generation_prompt=False,
    )
    return text


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def order_by_length(prompts: List[str]) -> Tuple[List[int], List[int]]:
    """
    Returns:
      order: indices sorted by prompt length (ascending)
      inv: inverse permutation so outputs can be restored to original order
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    inv = [0] * len(prompts)
    for new_pos, old_idx in enumerate(order):
        inv[old_idx] = new_pos
    return order, inv


# ---------------------------------------------------------------------------
# Postprocessing + fallback (guarantees non-empty)
# ---------------------------------------------------------------------------

_whitespace_re = re.compile(r"\s+")

def normalize_one_line(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def fallback_punchline(headline: str) -> str:
    h = normalize_one_line(headline)
    if h == "":
        return "I wrote a punchline, but it walked off stage without me."
    return f"Even '{h}' tried to be serious, but it could not keep a straight face."


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_batch_once(
    model,
    tokenizer,
    batch_prompts: List[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
) -> List[str]:
    texts = [build_chat_prompt(tokenizer, p) for p in batch_prompts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Critical: lengths as Python ints (CPU) BEFORE moving tensors to GPU
    input_lengths: List[int] = inputs["attention_mask"].sum(dim=1).tolist()

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=cfg.do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    outputs: List[str] = []
    for i in range(len(batch_prompts)):
        start = int(input_lengths[i])
        new_tokens = gen_ids[i, start:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(normalize_one_line(resp))

    return outputs


@torch.inference_mode()
def generate_responses_strict_non_empty(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
) -> List[str]:
    total = len(prompts)
    outputs_sorted: List[str] = [""] * total

    # Length bucketing = fewer padded tokens = less real GPU compute
    order, _ = order_by_length(prompts)
    start_all = time.time()

    for batch_index, batch_ids in enumerate(batched(order, batch_size), start=1):
        batch_prompts = [prompts[i] for i in batch_ids]

        print(f"\n[Batch {batch_index}] size={len(batch_ids)}")
        batch_out = generate_batch_once(
            model=model,
            tokenizer=tokenizer,
            batch_prompts=batch_prompts,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
        )

        for idx_in_batch, out_text in enumerate(batch_out):
            outputs_sorted[batch_ids[idx_in_batch]] = out_text

    # Retry only empties (keep retries minimal to reduce extra GPU passes)
    empty_indices = [i for i, x in enumerate(outputs_sorted) if normalize_one_line(x) == ""]
    print(f"\nInitial empty predictions: {len(empty_indices)}")

    for retry_round, (temp, tp, mx) in enumerate(cfg.retry_settings[: cfg.max_retries], start=1):
        if not empty_indices:
            break
        print(f"\nRetry round {retry_round}: temperature={temp}, top_p={tp}, max_new_tokens={mx}")

        retry_prompts = [prompts[i] for i in empty_indices]
        retry_out_all: List[str] = []

        retry_order, _ = order_by_length(retry_prompts)
        for chunk_ids in batched(retry_order, batch_size):
            chunk_prompts = [retry_prompts[j] for j in chunk_ids]
            retry_out_all.extend(
                generate_batch_once(
                    model=model,
                    tokenizer=tokenizer,
                    batch_prompts=chunk_prompts,
                    temperature=temp,
                    top_p=tp,
                    max_new_tokens=mx,
                    min_new_tokens=max(cfg.min_new_tokens, 8),
                )
            )

        retry_out = [""] * len(retry_prompts)
        for pos, j in enumerate(retry_order):
            retry_out[j] = retry_out_all[pos]

        still_empty: List[int] = []
        for orig_idx, new_text in zip(empty_indices, retry_out):
            new_text = normalize_one_line(new_text)
            if new_text != "":
                outputs_sorted[orig_idx] = new_text
            else:
                still_empty.append(orig_idx)

        empty_indices = still_empty
        print(f"Empty predictions after retry round {retry_round}: {len(empty_indices)}")

    # Final guarantee: never output empty
    if empty_indices:
        print(f"\nApplying fallback for remaining empties: {len(empty_indices)}")
        for idx in empty_indices:
            outputs_sorted[idx] = fallback_punchline(prompts[idx])

    final_empty = [i for i, x in enumerate(outputs_sorted) if normalize_one_line(x) == ""]
    if final_empty:
        raise RuntimeError(f"Non-empty guarantee failed at indices: {final_empty[:20]}")

    print(f"\nAll generations done in {time.time() - start_all:.1f}s total.")
    return outputs_sorted


# ---------------------------------------------------------------------------
# Input and output helpers
# ---------------------------------------------------------------------------

def save_and_zip(df_out: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    df_out = df_out.fillna("")
    print("Saving predictions to:", out_path)
    df_out.to_csv(out_path, sep="\t", index=False, na_rep="")

    zip_path = out_path.with_suffix(out_path.suffix + ".zip")
    print("Zipping to:", zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=out_path.name)

    return zip_path


def copy_to_drive(zip_path: Path, drive_dir: str):
    try:
        from google.colab import drive as colab_drive  # noqa: F401
    except ImportError:
        print("Not running in Colab, skipping Google Drive upload.")
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print(
            "Google Drive is not mounted. "
            "Run `from google.colab import drive; drive.mount('/content/drive')` first."
        )
        return

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
    configure_fast_kernels()

    print("Loading data from:", cfg.input_path)
    df = pd.read_csv(cfg.input_path, sep="\t", keep_default_na=False)

    if "headline" not in df.columns:
        raise ValueError(f"'headline' column not found in {cfg.input_path}")

    df["headline"] = df["headline"].fillna("").astype(str)
    prompts = df["headline"].tolist()
    print(f"Total prompts: {len(prompts)}")

    model, tokenizer = load_model_and_tokenizer()

    predictions = generate_responses_strict_non_empty(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=cfg.batch_size,
    )

    df_out = df.copy()
    df_out["prediction"] = predictions

    # Benchmark requirement: exactly zero empty cells
    empty_count = (df_out["prediction"].astype(str).map(normalize_one_line) == "").sum()
    if empty_count != 0:
        raise RuntimeError(f"Benchmark requirement failed: {empty_count} empty predictions")

    zip_path = save_and_zip(df_out, cfg.output_dir, cfg.output_filename)
    copy_to_drive(zip_path, cfg.drive_output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
