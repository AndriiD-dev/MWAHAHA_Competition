from __future__ import annotations

import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

import prompt_builder_title as pb


# ---------------------------------------------------------------------------
# Headline -> two nouns (anchors)
# ---------------------------------------------------------------------------

def _load_spacy_model():
    try:
        import spacy
    except Exception as e:
        raise RuntimeError(
            "spaCy is required. Install it and download the model:\n"
            "pip install spacy\n"
            "python -m spacy download en_core_web_sm"
        ) from e

    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(
            "spaCy model en_core_web_sm is missing.\n"
            "Run: python -m spacy download en_core_web_sm"
        ) from e


_NLP = None


def choose_two_nouns_from_label(
    gt_text_label: str,
    *,
    seed: int | None = None,
    prefer_distinct: bool = True,
) -> tuple[str, str]:
    """
    Takes text (here: a headline) and returns exactly two noun lemmas.

    - Extract NOUN / PROPN lemmas using spaCy.
    - Remove trivial lemmas.
    - Prefer distinct nouns; if not possible, duplicate the only one.
    - If no nouns, fall back to adjectives/verbs, then alphabetic tokens.
    """
    global _NLP
    if _NLP is None:
        _NLP = _load_spacy_model()

    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random

    text = (gt_text_label or "").strip()
    if not text:
        return ("", "")

    doc = _NLP(text)

    nouns: List[str] = []
    for t in doc:
        if t.pos_ in {"NOUN", "PROPN"}:
            lemma = t.lemma_.strip().lower()
            if lemma and lemma.isalpha() and len(lemma) > 2:
                nouns.append(lemma)

    junk = {
        "thing", "stuff", "something", "anything", "everything",
        "someone", "anyone", "everyone",
    }
    nouns = [n for n in nouns if n not in junk]

    if nouns:
        if prefer_distinct:
            unique = list(dict.fromkeys(nouns))
            if len(unique) >= 2:
                w1, w2 = rnd.sample(unique, 2)
                return (w1, w2)
            return (unique[0], unique[0])
        else:
            if len(nouns) >= 2:
                return tuple(rnd.sample(nouns, 2))  # type: ignore
            return (nouns[0], nouns[0])

    content: List[str] = []
    for t in doc:
        if t.pos_ in {"ADJ", "VERB"}:
            lemma = t.lemma_.strip().lower()
            if lemma and lemma.isalpha() and len(lemma) > 2:
                content.append(lemma)

    if content:
        unique = list(dict.fromkeys(content))
        if len(unique) >= 2:
            return tuple(rnd.sample(unique, 2))  # type: ignore
        return (unique[0], unique[0])

    tokens = [t.text.lower() for t in doc if t.text.isalpha() and len(t.text) > 2]
    if len(tokens) >= 2:
        return tuple(rnd.sample(tokens, 2))  # type: ignore
    if len(tokens) == 1:
        return (tokens[0], tokens[0])
    return ("", "")


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

    input_path: Path = DATA_DIR / "task-a-title.csv"
    output_dir: Path = OUTPUT_DIR
    output_filename: str = "task-a-title_predictions_base.tsv"

    # Deterministic noun selection
    noun_seed_base: int = 1337

    # Plan decoding (pass one)
    plan_max_new_tokens: int = 80
    plan_min_new_tokens: int = 16
    plan_temperature: float = 0.4
    plan_top_p: float = 0.9

    # Final decoding (pass two)
    max_new_tokens: int = 32
    min_new_tokens: int = 6
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    batch_size: int = 16

    # Retries for final jokes only
    max_retries: int = 20
    retry_settings: Tuple[Tuple[float, float, int], ...] = (
        (0.90, 0.95, 40),
        (0.95, 0.98, 40),
        (1.00, 0.98, 48),
        (1.05, 0.98, 48),
        (1.10, 0.99, 56),
        (1.15, 0.99, 64),
        (1.20, 0.99, 72),
        (1.25, 0.99, 80),
        (1.30, 1.00, 96),
        (1.35, 1.00, 96),
        (1.40, 1.00, 112),
        (1.45, 1.00, 112),
        (1.50, 1.00, 128),
        (1.55, 1.00, 128),
        (1.60, 1.00, 128),
        (1.65, 1.00, 128),
        (1.70, 1.00, 128),
        (1.75, 1.00, 128),
        (1.80, 1.00, 128),
        (1.85, 1.00, 128),
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)

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
        raise RuntimeError(f"Failed to load model. Last error: {last_err}")

    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    print("Model main device:", next(model.parameters()).device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Validation + fallbacks
# ---------------------------------------------------------------------------

def _boundary_pattern(phrase: str) -> re.Pattern:
    p = re.escape(pb.safe_word(phrase))
    return re.compile(rf"(?<![A-Za-z0-9]){p}(?![A-Za-z0-9])")


def anchor_nouns_present(text: str, noun1: str, noun2: str) -> bool:
    t = pb.normalize_one_line(text)
    if not t:
        return False
    n1 = pb.safe_word(noun1)
    n2 = pb.safe_word(noun2)
    if not n1 or not n2:
        return False
    p1 = _boundary_pattern(n1).search(t) is not None
    p2 = _boundary_pattern(n2).search(t) is not None
    return p1 and p2


def fallback_plan() -> str:
    return pb.normalize_one_line(
        '{"angle":"obvious headline irony","misdirection":"literal reading","word_placement":"use both nouns in the punchline","device":"reversal"}'
    )


def fallback_joke(headline: str, noun1: str, noun2: str) -> str:
    h = pb.normalize_one_line(headline)
    n1 = pb.safe_word(noun1) or "news"
    n2 = pb.safe_word(noun2) or "headline"
    if not h:
        return pb.normalize_one_line(f"My {n1} met a {n2}; somehow both still made the evening news.")
    return pb.normalize_one_line(f"After '{h}', my {n1} hired a {n2} for public relations. It went exactly as well as you think.")


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------

def to_chat_text(tokenizer, messages: List[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Correct boundary for left padding: generated tokens start after padded prompt width
    prompt_len = input_ids.shape[1]

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


def generate_all(
    model,
    tokenizer,
    chat_texts: List[str],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    min_new_tokens: int,
    label: str,
) -> List[str]:
    n = len(chat_texts)
    outputs: List[str] = [""] * n
    order = order_by_length(chat_texts)

    for batch_index, batch_ids in enumerate(batched(order, batch_size), start=1):
        batch_texts = [chat_texts[i] for i in batch_ids]
        print(f"\n[{label} batch {batch_index}] size={len(batch_ids)}")

        batch_out = generate_batch_once(
            model=model,
            tokenizer=tokenizer,
            chat_texts=batch_texts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )

        for j, out_text in enumerate(batch_out):
            outputs[batch_ids[j]] = out_text

    return outputs


@torch.inference_mode()
def generate_headline_predictions_with_plans(
    model,
    tokenizer,
    headlines: List[str],
    noun1_list: List[str],
    noun2_list: List[str],
    batch_size: int,
) -> List[str]:
    n = len(headlines)
    start_all = time.time()

    # Pass 1: plans for all
    plan_texts = [
        to_chat_text(tokenizer, pb.build_plan_messages(headlines[i], noun1_list[i], noun2_list[i]))
        for i in range(n)
    ]
    plans = generate_all(
        model=model,
        tokenizer=tokenizer,
        chat_texts=plan_texts,
        batch_size=batch_size,
        temperature=cfg.plan_temperature,
        top_p=cfg.plan_top_p,
        max_new_tokens=cfg.plan_max_new_tokens,
        min_new_tokens=cfg.plan_min_new_tokens,
        label="PLAN",
    )
    plans = [p if pb.normalize_one_line(p) else fallback_plan() for p in plans]

    # Pass 2: final jokes for all
    final_texts = [
        to_chat_text(tokenizer, pb.build_final_messages(headlines[i], noun1_list[i], noun2_list[i], plans[i]))
        for i in range(n)
    ]
    outputs = generate_all(
        model=model,
        tokenizer=tokenizer,
        chat_texts=final_texts,
        batch_size=batch_size,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        min_new_tokens=cfg.min_new_tokens,
        label="FINAL",
    )

    def is_good(i: int) -> bool:
        t = outputs[i]
        if pb.normalize_one_line(t) == "":
            return False
        return anchor_nouns_present(t, noun1_list[i], noun2_list[i])

    bad_indices = [i for i in range(n) if not is_good(i)]
    print(f"\nInitial final failures (empty or missing anchor nouns): {len(bad_indices)}")

    strict_suffix = (
        "IMPORTANT: The answer is INVALID unless it includes BOTH anchor nouns exactly as written. "
        "Output only the joke."
    )

    # Retry final only (keep plan fixed)
    for retry_round, (temp, tp, mx) in enumerate(cfg.retry_settings[: cfg.max_retries], start=1):
        if not bad_indices:
            break

        print(f"\nFINAL retry round {retry_round}: temperature={temp}, top_p={tp}, max_new_tokens={mx}")

        retry_texts = []
        for i in bad_indices:
            msgs = pb.build_final_messages(headlines[i], noun1_list[i], noun2_list[i], plans[i])
            msgs = [dict(msgs[0]), dict(msgs[1])]
            msgs[1]["content"] = pb.normalize_one_line(msgs[1]["content"] + "\n" + strict_suffix)
            retry_texts.append(to_chat_text(tokenizer, msgs))

        retry_out = generate_all(
            model=model,
            tokenizer=tokenizer,
            chat_texts=retry_texts,
            batch_size=batch_size,
            temperature=temp,
            top_p=tp,
            max_new_tokens=mx,
            min_new_tokens=max(cfg.min_new_tokens, 8),
            label=f"FINAL RETRY {retry_round}",
        )

        new_bad: List[int] = []
        for pos, i in enumerate(bad_indices):
            candidate = pb.normalize_one_line(retry_out[pos])
            if candidate and anchor_nouns_present(candidate, noun1_list[i], noun2_list[i]):
                outputs[i] = candidate
            else:
                new_bad.append(i)

        bad_indices = new_bad
        print(f"Final failures after retry round {retry_round}: {len(bad_indices)}")

    # Final guarantee
    if bad_indices:
        print(f"\nApplying fallback for remaining failures: {len(bad_indices)}")
        for i in bad_indices:
            outputs[i] = fallback_joke(headlines[i], noun1_list[i], noun2_list[i])

    still_bad = [
        i for i in range(n)
        if pb.normalize_one_line(outputs[i]) == "" or not anchor_nouns_present(outputs[i], noun1_list[i], noun2_list[i])
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
    except Exception:
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
    # Your earlier scripts read task-a-title.csv using tab separator
    df = pd.read_csv(cfg.input_path, sep="\t", keep_default_na=False)

    if "headline" not in df.columns:
        raise ValueError(f"Missing required column 'headline' in {cfg.input_path}")

    df["headline"] = df["headline"].fillna("").astype(str)
    headlines = df["headline"].tolist()
    print(f"Total rows: {len(headlines)}")

    noun1_list: List[str] = []
    noun2_list: List[str] = []
    for i, h in enumerate(headlines):
        seed = cfg.noun_seed_base + i
        n1, n2 = choose_two_nouns_from_label(h, seed=seed, prefer_distinct=True)
        noun1_list.append(n1)
        noun2_list.append(n2)

    model, tokenizer = load_model_and_tokenizer()

    predictions = generate_headline_predictions_with_plans(
        model=model,
        tokenizer=tokenizer,
        headlines=headlines,
        noun1_list=noun1_list,
        noun2_list=noun2_list,
        batch_size=cfg.batch_size,
    )

    df_out = df.copy()
    # Optional: keep anchors for debugging. Remove if submission format is strict.
    df_out["noun1"] = noun1_list
    df_out["noun2"] = noun2_list
    df_out["prediction"] = predictions

    empty_count = (df_out["prediction"].astype(str).map(pb.normalize_one_line) == "").sum()
    if empty_count != 0:
        raise RuntimeError(f"Requirement failed: {empty_count} empty predictions")

    zip_path = save_and_zip(df_out, cfg.output_dir, cfg.output_filename)
    copy_to_drive(zip_path, cfg.drive_output_dir)

    print("All done.")


if __name__ == "__main__":
    main()
