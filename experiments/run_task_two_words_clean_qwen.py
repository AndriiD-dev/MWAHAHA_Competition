from __future__ import annotations

import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import torch
import pandas as pd
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
# Config
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"

    input_path: Path = DATA_DIR / "task-a-two-words.csv"
    output_dir: Path = OUTPUT_DIR
    output_filename: str = "task-a-two-words_predictions_base.tsv"

    # Decoding
    max_new_tokens: int = 32
    min_new_tokens: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    # Throughput
    batch_size: int = 16

    # “Zero empty cells” + “must include both words” policy
    max_retries: int = 2
    retry_settings: Tuple[Tuple[float, float, int], ...] = (
        (0.95, 0.98, 40),
        (1.05, 0.98, 48),
    )

    drive_output_dir: str = "/content/drive/MyDrive/MWAHAHA_outputs"


cfg = InferenceConfig()

SYSTEM_PROMPT = (
    "You are a stand-up comedian. Write ONE original joke in English. "
    "Return exactly one line under 30 words. No preface, no explanation, no emojis. "
    "Do not output any chat markers, speaker labels, or formatting headers. Output only the joke text. "
    "Avoid hate, slurs, explicit sex, and graphic violence."
)


# ---------------------------------------------------------------------------
# Speed knobs
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
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print("Loading tokenizer from base model:", cfg.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
        print("Using CUDA (float16)")
    else:
        torch_dtype = torch.float32
        device_map = "auto"
        print("CUDA not available, using CPU (float32)")

    attn_impl_candidates = ["flash_attention_2", "sdpa", "eager"]
    last_err = None
    model = None

    for attn_impl in attn_impl_candidates:
        try:
            print(f"Loading base model with attention implementation: {attn_impl}")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            break
        except Exception as e:
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(f"Failed to load model with any attention implementation. Last error: {last_err}")

    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    print("Model main device:", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_whitespace_re = re.compile(r"\s+")

def normalize_one_line(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def safe_word(w: str) -> str:
    return normalize_one_line(w).strip()


def make_user_prompt_from_words(word1: str, word2: str) -> str:
    w1 = safe_word(word1)
    w2 = safe_word(word2)
    return (
        f"Write one joke that MUST include the two words exactly as written: '{w1}' and '{w2}'. "
        "Use both words as normal words in the joke. One line only."
    )


def make_retry_prompt_from_words(word1: str, word2: str) -> str:
    w1 = safe_word(word1)
    w2 = safe_word(word2)
    return (
        f"Write one joke. REQUIRED: include '{w1}' and '{w2}' exactly. "
        "If you miss either word, the answer is wrong. One line only."
    )


def required_words_present(text: str, word1: str, word2: str) -> bool:
    t = normalize_one_line(text).lower()
    w1 = re.escape(safe_word(word1).lower())
    w2 = re.escape(safe_word(word2).lower())

    # whole-word match
    p1 = re.search(rf"\b{w1}\b", t) is not None
    p2 = re.search(rf"\b{w2}\b", t) is not None
    return p1 and p2


def fallback_joke(word1: str, word2: str) -> str:
    # Guaranteed inclusion, one line, short.
    w1 = safe_word(word1)
    w2 = safe_word(word2)
    return normalize_one_line(f"I brought a {w1} to a {w2} meeting; both were unqualified, but somehow still got promoted.")


# ---------------------------------------------------------------------------
# Chat prompt
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


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i: i + batch_size]


def order_by_length(prompts: List[str]) -> List[int]:
    return sorted(range(len(prompts)), key=lambda i: len(prompts[i]))


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

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Correct boundary for left-padded batches
    prompt_len = input_ids.shape[1]

    # Block common chat markers / role-like strings
    BAD_PHRASES = [
        "assistant", "assistant:", "user", "user:", "system", "system:",
        "assistant user", "<tool call>", "</tool call>",
    ]
    bad_words_ids = tokenizer(BAD_PHRASES, add_special_tokens=False).input_ids

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
        bad_words_ids=bad_words_ids,
    )

    new_tokens = gen_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [normalize_one_line(x) for x in decoded]


@torch.inference_mode()
def generate_word_inclusion_predictions(
    model,
    tokenizer,
    word1_list: List[str],
    word2_list: List[str],
    batch_size: int,
) -> List[str]:
    n = len(word1_list)
    prompts = [make_user_prompt_from_words(word1_list[i], word2_list[i]) for i in range(n)]

    outputs: List[str] = [""] * n
    order = order_by_length(prompts)
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

        for j, out_text in enumerate(batch_out):
            outputs[batch_ids[j]] = out_text

    def is_good(i: int) -> bool:
        t = outputs[i]
        if normalize_one_line(t) == "":
            return False
        return required_words_present(t, word1_list[i], word2_list[i])

    bad_indices = [i for i in range(n) if not is_good(i)]
    print(f"\nInitial failures (empty or missing required words): {len(bad_indices)}")

    for retry_round, (temp, tp, mx) in enumerate(cfg.retry_settings[: cfg.max_retries], start=1):
        if not bad_indices:
            break

        print(f"\nRetry round {retry_round}: temperature={temp}, top_p={tp}, max_new_tokens={mx}")

        retry_prompts = [make_retry_prompt_from_words(word1_list[i], word2_list[i]) for i in bad_indices]
        retry_order = order_by_length(retry_prompts)

        retry_out_all: List[str] = []
        for chunk_ids in batched(retry_order, batch_size):
            chunk_prompts = [retry_prompts[k] for k in chunk_ids]
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

        # Map retry outputs back to bad_indices
        retry_out = [""] * len(retry_prompts)
        for pos, k in enumerate(retry_order):
            retry_out[k] = retry_out_all[pos]

        new_bad: List[int] = []
        for idx_pos, original_i in enumerate(bad_indices):
            candidate = normalize_one_line(retry_out[idx_pos])
            if candidate != "" and required_words_present(candidate, word1_list[original_i], word2_list[original_i]):
                outputs[original_i] = candidate
            else:
                new_bad.append(original_i)

        bad_indices = new_bad
        print(f"Failures after retry round {retry_round}: {len(bad_indices)}")

    # Final guarantee: valid and non-empty
    if bad_indices:
        print(f"\nApplying fallback for remaining failures: {len(bad_indices)}")
        for i in bad_indices:
            outputs[i] = fallback_joke(word1_list[i], word2_list[i])

    # Assert compliance
    still_bad = [i for i in range(n) if normalize_one_line(outputs[i]) == "" or not required_words_present(outputs[i], word1_list[i], word2_list[i])]
    if still_bad:
        raise RuntimeError(f"Validation failed for indices: {still_bad[:20]}")

    print(f"\nAll generations done in {time.time() - start_all:.1f}s total.")
    return outputs


# ---------------------------------------------------------------------------
# I/O helpers
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

    required_cols = {"word1", "word2"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {cfg.input_path}: {sorted(missing)}")

    df["word1"] = df["word1"].fillna("").astype(str)
    df["word2"] = df["word2"].fillna("").astype(str)

    word1_list = df["word1"].tolist()
    word2_list = df["word2"].tolist()
    print(f"Total rows: {len(df)}")

    model, tokenizer = load_model_and_tokenizer()

    predictions = generate_word_inclusion_predictions(
        model=model,
        tokenizer=tokenizer,
        word1_list=word1_list,
        word2_list=word2_list,
        batch_size=cfg.batch_size,
    )

    df_out = df.copy()
    df_out["prediction"] = predictions

    # Final validation
    empty_count = (df_out["prediction"].astype(str).map(normalize_one_line) == "").sum()
    if empty_count != 0:
        raise RuntimeError(f"Requirement failed: {empty_count} empty predictions")

    missing_words_count = 0
    for w1, w2, pred in zip(word1_list, word2_list, df_out["prediction"].tolist()):
        if not required_words_present(pred, w1, w2):
            missing_words_count += 1
    if missing_words_count != 0:
        raise RuntimeError(f"Requirement failed: {missing_words_count} predictions missing required words")

    zip_path = save_and_zip(df_out, cfg.output_dir, cfg.output_filename)
    copy_to_drive(zip_path, cfg.drive_output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
