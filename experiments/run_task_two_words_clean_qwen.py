from __future__ import annotations

import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

import prompt_builder as pb


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

    # “Zero empty cells” + “must include both words”
    max_retries: int = 2
    retry_settings: Tuple[Tuple[float, float, int], ...] = (
        (0.95, 0.98, 40),
        (1.05, 0.98, 48),
    )

    drive_output_dir: str = "/content/drive/MyDrive/MWAHAHA_outputs"


cfg = InferenceConfig()


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

    # Decoder-only models: left padding is safer for batched generation
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
# Validation helpers
# ---------------------------------------------------------------------------

def _boundary_pattern(phrase: str) -> re.Pattern:
    """
    Match the phrase as a standalone token/phrase, allowing punctuation around it.
    Uses alphanumeric boundaries (works better than \\b for hyphens/spaces).
    """
    p = re.escape(pb.safe_word(phrase))
    return re.compile(rf"(?<![A-Za-z0-9]){p}(?![A-Za-z0-9])")

def required_words_present(text: str, word1: str, word2: str) -> bool:
    t = pb.normalize_one_line(text)
    if not t:
        return False
    w1 = pb.safe_word(word1)
    w2 = pb.safe_word(word2)
    if not w1 or not w2:
        return False

    p1 = _boundary_pattern(w1).search(t) is not None
    p2 = _boundary_pattern(w2).search(t) is not None
    return p1 and p2


def fallback_joke(word1: str, word2: str) -> str:
    # Guaranteed inclusion, one line, short.
    w1 = pb.safe_word(word1)
    w2 = pb.safe_word(word2)
    return pb.normalize_one_line(
        f"I brought a {w1} to a {w2} meeting; both were unqualified, but somehow still got promoted."
    )


# ---------------------------------------------------------------------------
# Prompt building (delegated to prompt_builder.py)
# ---------------------------------------------------------------------------

def build_chat_text(tokenizer, word1: str, word2: str, strict_suffix: str = "") -> str:
    """
    Uses prompt_builder.build_joke_messages() and converts to a model-ready chat template string.
    strict_suffix optionally appends extra instruction for retries.
    """
    messages = pb.build_joke_messages(word1, word2)
    if strict_suffix:
        # Append extra constraint to the user message without re-implementing the builder logic.
        messages = [dict(messages[0]), dict(messages[1])]
        messages[1]["content"] = pb.normalize_one_line(messages[1]["content"] + "\n" + strict_suffix)

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
        yield items[i : i + batch_size]


def order_by_length(texts: List[str]) -> List[int]:
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_batch_once(
    model,
    tokenizer,
    chat_texts: List[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
) -> List[str]:
    inputs = tokenizer(
        chat_texts,
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
    return [pb.normalize_one_line(x) for x in decoded]


@torch.inference_mode()
def generate_word_inclusion_predictions(
    model,
    tokenizer,
    word1_list: List[str],
    word2_list: List[str],
    batch_size: int,
) -> List[str]:
    n = len(word1_list)

    # Build initial chat texts (includes micro-cards/facts from prompt_builder)
    chat_texts = [
        build_chat_text(tokenizer, word1_list[i], word2_list[i])
        for i in range(n)
    ]

    outputs: List[str] = [""] * n
    order = order_by_length(chat_texts)
    start_all = time.time()

    for batch_index, batch_ids in enumerate(batched(order, batch_size), start=1):
        batch_texts = [chat_texts[i] for i in batch_ids]
        print(f"\n[Batch {batch_index}] size={len(batch_ids)}")

        batch_out = generate_batch_once(
            model=model,
            tokenizer=tokenizer,
            chat_texts=batch_texts,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
        )

        for j, out_text in enumerate(batch_out):
            outputs[batch_ids[j]] = out_text

    def is_good(i: int) -> bool:
        t = outputs[i]
        if pb.normalize_one_line(t) == "":
            return False
        return required_words_present(t, word1_list[i], word2_list[i])

    bad_indices = [i for i in range(n) if not is_good(i)]
    print(f"\nInitial failures (empty or missing required words): {len(bad_indices)}")

    strict_suffix = "IMPORTANT: The joke is INVALID unless it includes BOTH required words exactly as written."

    for retry_round, (temp, tp, mx) in enumerate(cfg.retry_settings[: cfg.max_retries], start=1):
        if not bad_indices:
            break

        print(f"\nRetry round {retry_round}: temperature={temp}, top_p={tp}, max_new_tokens={mx}")

        retry_texts = [
            build_chat_text(tokenizer, word1_list[i], word2_list[i], strict_suffix=strict_suffix)
            for i in bad_indices
        ]
        retry_order = order_by_length(retry_texts)

        retry_out_all: List[str] = []
        for chunk_ids in batched(retry_order, batch_size):
            chunk_texts = [retry_texts[k] for k in chunk_ids]
            retry_out_all.extend(
                generate_batch_once(
                    model=model,
                    tokenizer=tokenizer,
                    chat_texts=chunk_texts,
                    temperature=temp,
                    top_p=tp,
                    max_new_tokens=mx,
                    min_new_tokens=max(cfg.min_new_tokens, 8),
                )
            )

        # Map retry outputs back to bad_indices
        retry_out = [""] * len(retry_texts)
        for pos, k in enumerate(retry_order):
            retry_out[k] = retry_out_all[pos]

        new_bad: List[int] = []
        for idx_pos, original_i in enumerate(bad_indices):
            candidate = pb.normalize_one_line(retry_out[idx_pos])
            if candidate and required_words_present(candidate, word1_list[original_i], word2_list[original_i]):
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

    still_bad = [
        i for i in range(n)
        if pb.normalize_one_line(outputs[i]) == "" or not required_words_present(outputs[i], word1_list[i], word2_list[i])
    ]
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
    empty_count = (df_out["prediction"].astype(str).map(pb.normalize_one_line) == "").sum()
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
