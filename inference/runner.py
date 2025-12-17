from __future__ import annotations

# NOTE:
# This module is executed as: python -m inference.runner --config <path/to/config.py>
# Keep all environment toggles BEFORE importing transformers to avoid TensorFlow/absl noise.

import os

# Silence (and avoid importing) TensorFlow when using Hugging Face Transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.utils.prompt_builder import PromptBuilder, normalize_one_line, safe_word
from inference.utils.response_evaluator import ResponseEvaluator
from inference.utils.logger import Logger


# =============================================================================
# Runner configuration
# =============================================================================


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int
    min_new_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True)
class RetryStep:
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class RunnerConfig:
    # Task selector: "two_words" (Task A, Two Words) or "title" (Task A, Headline)
    task: str

    base_model_id: str
    input_path: Path
    output_dir: Path
    output_filename: str

    batch_size: int = 16

    # Plan stage decoding
    plan_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=80,
        min_new_tokens=16,
        temperature=0.4,
        top_p=0.9,
    )

    # Final stage decoding
    final_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=64,
        min_new_tokens=10,
        temperature=0.9,
        top_p=0.95,
    )

    max_retries: int = 5
    retry_steps: Tuple[RetryStep, ...] = (
        RetryStep(0.90, 0.95, 40),
        RetryStep(0.95, 0.98, 40),
        RetryStep(1.00, 0.98, 48),
        RetryStep(1.05, 0.98, 48),
        RetryStep(1.10, 0.99, 56),
    )

    # Title-only knobs
    noun_seed_base: int = 42
    replan_every: int = 3  # 0 disables replanning for title task

    # Optional: copy outputs (zip + log) to Google Drive
    drive_output_dir: Optional[str] = None


def _as_path(x: Any) -> Path:
    if isinstance(x, Path):
        return x
    return Path(str(x))


def _decode_from_dict(d: Dict[str, Any]) -> DecodeConfig:
    return DecodeConfig(
        max_new_tokens=int(d.get("max_new_tokens", 64)),
        min_new_tokens=int(d.get("min_new_tokens", 0)),
        temperature=float(d.get("temperature", 1.0)),
        top_p=float(d.get("top_p", 1.0)),
    )


def _retry_steps_from_list(xs: Sequence[Dict[str, Any]]) -> Tuple[RetryStep, ...]:
    out: List[RetryStep] = []
    for d in xs:
        out.append(
            RetryStep(
                temperature=float(d.get("temperature", 1.0)),
                top_p=float(d.get("top_p", 1.0)),
                max_new_tokens=int(d.get("max_new_tokens", 48)),
            )
        )
    return tuple(out)




def load_runner_config(path: Path) -> RunnerConfig:
    """
    Loads a config python file that defines RUNNER_CONFIG as either:
    - a dict (recommended), or
    - a RunnerConfig instance.
    """
    spec = importlib.util.spec_from_file_location("runner_config_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if not hasattr(mod, "RUNNER_CONFIG"):
        raise AttributeError("Config file must define RUNNER_CONFIG")

    cfg_obj = getattr(mod, "RUNNER_CONFIG")

    if isinstance(cfg_obj, RunnerConfig):
        return cfg_obj

    if not isinstance(cfg_obj, dict):
        raise TypeError("RUNNER_CONFIG must be RunnerConfig or dict")

    d = dict(cfg_obj)

    # Required
    task = str(d["task"])
    base_model_id = str(d["base_model_id"])
    input_path = _as_path(d["input_path"])
    output_dir = _as_path(d["output_dir"])
    output_filename = str(d["output_filename"])

    # Optional
    batch_size = int(d.get("batch_size", 16))
    noun_seed_base = int(d.get("noun_seed_base", 1337))
    replan_every = int(d.get("replan_every", 3))
    max_retries = int(d.get("max_retries", 5))
    drive_output_dir = d.get("drive_output_dir")

    plan_decode = _decode_from_dict(d.get("plan_decode", {}))
    final_decode = _decode_from_dict(d.get("final_decode", {}))

    retry_steps_cfg = d.get("retry_steps")
    if retry_steps_cfg is None:
        retry_steps = RunnerConfig.retry_steps
    else:
        retry_steps = _retry_steps_from_list(retry_steps_cfg)

    return RunnerConfig(
        task=task,
        base_model_id=base_model_id,
        input_path=input_path,
        output_dir=output_dir,
        output_filename=output_filename,
        batch_size=batch_size,
        plan_decode=plan_decode,
        final_decode=final_decode,
        max_retries=max_retries,
        retry_steps=retry_steps,
        noun_seed_base=noun_seed_base,
        replan_every=replan_every,
        drive_output_dir=drive_output_dir,
    )


# =============================================================================
# Utilities
# =============================================================================


def configure_fast_kernels() -> None:
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def pick_dtype() -> torch.dtype:
    """
    Choose a safe dtype for Colab GPUs:
    - bfloat16 only on newer GPUs
    - float16 otherwise
    """
    if not torch.cuda.is_available():
        return torch.float32

    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

    # Decoder-only models should use left padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=dtype if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Make generation robust with padding.
    try:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    model.eval()
    return model, tokenizer


def to_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def batched(indices: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    batch: List[int] = []
    for i in indices:
        batch.append(i)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_batch_once(model, tokenizer, chat_texts: List[str], decode: DecodeConfig) -> List[str]:
    if not chat_texts:
        return []

    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True)

    target_device = getattr(model, "device", None)
    if target_device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=decode.max_new_tokens,
        min_new_tokens=decode.min_new_tokens,
        do_sample=True,
        temperature=float(decode.temperature),
        top_p=float(decode.top_p),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Slice off the prompt part (same length for all because of padding).
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_only = gen[:, prompt_len:]
    outs = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return [normalize_one_line(x) for x in outs]


def zip_file(path: Path) -> Path:
    zip_path = path.with_suffix(path.suffix + ".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, arcname=path.name)
    return zip_path


def copy_outputs_to_drive(paths: Sequence[Path], drive_dir: Optional[str]) -> None:
    if not drive_dir:
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("Drive not mounted; skipping copy to Drive.")
        return

    drive_dir_path = Path(drive_dir)
    if drive_dir_path.is_absolute():
        try:
            rel = drive_dir_path.relative_to(drive_root)
            dest_dir = drive_root / rel
        except Exception:
            dest_dir = drive_dir_path
    else:
        dest_dir = drive_root / drive_dir_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    for p in paths:
        if p.exists():
            shutil.copy2(p, dest_dir / p.name)


# =============================================================================
# Runner class (handles both tasks)
# =============================================================================


@dataclass
class InferenceRunner:
    cfg: RunnerConfig
    builder: PromptBuilder = field(default_factory=PromptBuilder)

    # Create ResponseEvaluator lazily so we can print progress and avoid "looks stuck".
    response_evaluator: Optional[ResponseEvaluator] = field(default=None, init=False)

    def _ensure_evaluator(self) -> ResponseEvaluator:
        if self.response_evaluator is None:
            print("Initializing ResponseEvaluator (required-words checker + humor classifier)...")
            self.response_evaluator = ResponseEvaluator()
        return self.response_evaluator

    def _load_dataframe(self) -> pd.DataFrame:
        if self.cfg.task == "two_words":
            df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
            need = {"word1", "word2"}
            missing = need.difference(set(df.columns))
            if missing:
                raise ValueError(f"Missing required columns in {self.cfg.input_path}: {sorted(missing)}")
            df["word1"] = df["word1"].fillna("").astype(str)
            df["word2"] = df["word2"].fillna("").astype(str)
            return df

        if self.cfg.task == "title":
            # Title files in this project are typically TSV (tab-separated), even if named *.csv.
            # Using comma-sep breaks when headlines contain commas.
            try:
                df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
            except Exception:
                # Last-resort: treat as "one headline per line"
                lines = Path(self.cfg.input_path).read_text(encoding="utf-8", errors="replace").splitlines()
                # drop header if present
                if lines and lines[0].strip().lower() == "headline":
                    lines = lines[1:]
                df = pd.DataFrame({"headline": lines})

            # If the file has no header row, pandas might name the column 0
            if "headline" not in df.columns:
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: "headline"})
                elif len(df.columns) >= 2:
                    # common case: id + headline
                    df = df.rename(columns={df.columns[1]: "headline"})

            if "headline" not in df.columns:
                raise ValueError(f"Missing required column 'headline' in {self.cfg.input_path}")

            df["headline"] = df["headline"].fillna("").astype(str)

            print("Preparing nouns for each headline (spaCy noun extraction)...")
            # Force spaCy model load once so the first extraction does not look like a freeze.
            try:
                self.builder._ensure_models()  # noqa: SLF001
            except Exception:
                pass

            noun1_list: List[str] = []
            noun2_list: List[str] = []
            n = len(df)

            t0 = time.time()
            for i in range(n):
                headline = df.loc[i, "headline"]
                seed = int(self.cfg.noun_seed_base + i)
                n1, n2 = self.builder.extract_title_nouns(headline, seed=seed)
                noun1_list.append(n1)
                noun2_list.append(n2)

                if (i + 1) % 50 == 0 or (i + 1) == n:
                    dt = time.time() - t0
                    print(f"  nouns: {i + 1}/{n}  (elapsed {dt:.1f}s)")

            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df

        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _anchors_for_row(self, df: pd.DataFrame, i: int) -> Tuple[str, str]:
        if self.cfg.task == "two_words":
            return str(df.loc[i, "word1"]), str(df.loc[i, "word2"])
        return str(df.loc[i, "noun1"]), str(df.loc[i, "noun2"])

    def _fallback_plan(self, df: pd.DataFrame, i: int) -> str:
        w1, w2 = self._anchors_for_row(df, i)
        w1 = safe_word(w1) or w1
        w2 = safe_word(w2) or w2
        return json.dumps(
            {
                "scenario": f"A simple everyday situation involving {w1}.",
                "misdirection": f"Set up a normal expectation, then twist it with {w2}.",
                "word_placement": f"Use '{w1}' early and '{w2}' late.",
                "device": "wordplay",
            },
            ensure_ascii=False,
        )

    def _fallback_output(self, df: pd.DataFrame, i: int) -> str:
        w1, w2 = self._anchors_for_row(df, i)
        w1 = safe_word(w1) or w1
        w2 = safe_word(w2) or w2
        return normalize_one_line(f"I tried to {w1} my life together, but my {w2} had other plans.")

    def _build_plan_messages(self, df: pd.DataFrame, i: int) -> List[Dict[str, str]]:
        if self.cfg.task == "two_words":
            return self.builder.build_two_words_plan_messages(df.loc[i, "word1"], df.loc[i, "word2"])
        return self.builder.build_title_plan_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"])

    def _build_final_messages(self, df: pd.DataFrame, i: int, plan_json: str) -> List[Dict[str, str]]:
        if self.cfg.task == "two_words":
            return self.builder.build_two_words_final_messages(df.loc[i, "word1"], df.loc[i, "word2"], plan_json)
        return self.builder.build_title_final_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"], plan_json)

    def _eval_dict(self, out_text: str, word1: str, word2: str) -> Dict[str, Any]:
        evaluator = self._ensure_evaluator()
        required_ok = evaluator.required_words_present(out_text, word1, word2)
        humorous_ok = evaluator.is_humorous(out_text)
        good_ok = bool(required_ok and humorous_ok)
        return {
            "required_words": {
                "anchor1": word1,
                "anchor2": word2,
                "present": bool(required_ok),
            },
            "humor_classifier": {
                "humorous": bool(humorous_ok),
            },
            "overall": {
                "good": good_ok,
            },
        }

    def _is_good(self, df: pd.DataFrame, i: int, out: str) -> bool:
        out = normalize_one_line(out)
        if out == "":
            return False
        word1, word2 = self._anchors_for_row(df, i)
        return bool(self._eval_dict(out, word1, word2)["overall"]["good"])

    def run(self) -> None:
        configure_fast_kernels()

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cfg.output_dir / self.cfg.output_filename
        log_path = out_path.with_suffix(out_path.suffix + ".log.json")

        print("Project root:", self.builder.config.paths.project_root)
        print("Using input :", self.cfg.input_path)
        print("Using output:", out_path)
        print("Using log   :", log_path)

        # Heavy pre-processing first (so "Loading checkpoint shards" is not followed by a long silence).
        df = self._load_dataframe()
        n = len(df)
        print(f"Loaded {n} rows.")

        # Logger opens file immediately and writes "run_start".
        logger = Logger(enabled=True, log_path=log_path, max_chars=900)

        logger.log_run_meta(
            {
                "task": self.cfg.task,
                "base_model_id": self.cfg.base_model_id,
                "input_path": str(self.cfg.input_path),
                "output_dir": str(self.cfg.output_dir),
                "output_filename": self.cfg.output_filename,
                "batch_size": self.cfg.batch_size,
                "max_retries": self.cfg.max_retries,
                "replan_every": self.cfg.replan_every,
                "drive_output_dir": self.cfg.drive_output_dir,
            }
        )

        # Now load the main model.
        print("Loading base model + tokenizer...")
        model, tokenizer = load_model_and_tokenizer(self.cfg.base_model_id)

        # Per-row metadata for logging.
        metas: List[Dict[str, Any]] = []
        for i in range(n):
            meta: Dict[str, Any] = {"anchors": {"word1": "", "word2": ""}, "headline": None, "plan": None}
            if self.cfg.task == "two_words":
                meta["anchors"]["word1"] = str(df.loc[i, "word1"])
                meta["anchors"]["word2"] = str(df.loc[i, "word2"])
            else:
                meta["headline"] = str(df.loc[i, "headline"])
                meta["anchors"]["word1"] = str(df.loc[i, "noun1"])
                meta["anchors"]["word2"] = str(df.loc[i, "noun2"])
            metas.append(meta)

        indices = list(range(n))

        # ---------------------------------------------------------------------
        # PLAN stage
        # ---------------------------------------------------------------------
        print("\nStage: PLAN")
        plans: List[str] = [""] * n
        t_plan = time.time()

        for batch_index, batch_ids in enumerate(batched(indices, self.cfg.batch_size), start=1):
            batch_texts: List[str] = []
            for i in batch_ids:
                msgs = self._build_plan_messages(df, i)
                chat = to_chat_text(tokenizer, msgs)
                batch_texts.append(chat)

            batch_out = generate_batch_once(model, tokenizer, batch_texts, self.cfg.plan_decode)

            for local, i in enumerate(batch_ids):
                p = normalize_one_line(batch_out[local])
                if not p:
                    p = self._fallback_plan(df, i)
                plans[i] = p
                metas[i]["plan"] = logger.try_parse_json(p) or p

            if batch_ids:
                logger.log_first_in_batch(
                    stage="PLAN",
                    batch_index=batch_index,
                    batch_size=len(batch_ids),
                    example_index=batch_ids[0],
                    chat_text=batch_texts[0],
                    model_output=batch_out[0] if batch_out else "",
                    meta=metas[batch_ids[0]],
                )

            if batch_index % 10 == 0:
                dt = time.time() - t_plan
                done = min(batch_index * self.cfg.batch_size, n)
                print(f"  plan batches: {batch_index}  ({done}/{n}, elapsed {dt:.1f}s)")

        print(f"PLAN done in {time.time() - t_plan:.1f}s")

        # ---------------------------------------------------------------------
        # FINAL stage (+ retries + fallback)
        # ---------------------------------------------------------------------
        print("\nStage: FINAL")
        outputs: List[str] = [""] * n
        used_fallback: List[bool] = [False] * n
        last_before_fallback: List[str] = [""] * n
        ever_failed: set[int] = set()

        t_final = time.time()
        for batch_index, batch_ids in enumerate(batched(indices, self.cfg.batch_size), start=1):
            batch_texts: List[str] = []
            for i in batch_ids:
                msgs = self._build_final_messages(df, i, plans[i])
                batch_texts.append(to_chat_text(tokenizer, msgs))

            batch_out = generate_batch_once(model, tokenizer, batch_texts, self.cfg.final_decode)

            for local, i in enumerate(batch_ids):
                out = normalize_one_line(batch_out[local])
                outputs[i] = out
                last_before_fallback[i] = out

            if batch_ids:
                i0 = batch_ids[0]
                w1, w2 = self._anchors_for_row(df, i0)
                evaluation = None
                try:
                    evaluation = self._eval_dict(outputs[i0], w1, w2)
                except Exception:
                    evaluation = None

                logger.log_first_in_batch(
                    stage="FINAL",
                    batch_index=batch_index,
                    batch_size=len(batch_ids),
                    example_index=i0,
                    chat_text=batch_texts[0],
                    model_output=batch_out[0] if batch_out else "",
                    meta=metas[i0],
                    evaluation=evaluation,
                )

            if batch_index % 10 == 0:
                dt = time.time() - t_final
                done = min(batch_index * self.cfg.batch_size, n)
                print(f"  final batches: {batch_index}  ({done}/{n}, elapsed {dt:.1f}s)")

        bad_indices: List[int] = []
        for i in indices:
            if not self._is_good(df, i, outputs[i]):
                bad_indices.append(i)
                ever_failed.add(i)

        for retry_round in range(1, self.cfg.max_retries + 1):
            if not bad_indices:
                break

            step = self.cfg.retry_steps[(retry_round - 1) % len(self.cfg.retry_steps)]
            decode = DecodeConfig(
                max_new_tokens=step.max_new_tokens,
                min_new_tokens=max(6, self.cfg.final_decode.min_new_tokens),
                temperature=step.temperature,
                top_p=step.top_p,
            )

            print(
                f"\nRetry {retry_round}: regenerating {len(bad_indices)} failures "
                f"(temperature={decode.temperature}, top_p={decode.top_p})"
            )

            if self.cfg.task == "title" and self.cfg.replan_every > 0 and (retry_round % self.cfg.replan_every == 0):
                print(f"  Replanning {len(bad_indices)} items...")
                for batch_index, batch_ids in enumerate(batched(bad_indices, self.cfg.batch_size), start=1):
                    batch_texts: List[str] = []
                    for i in batch_ids:
                        msgs = self._build_plan_messages(df, i)
                        batch_texts.append(to_chat_text(tokenizer, msgs))
                    batch_out = generate_batch_once(model, tokenizer, batch_texts, self.cfg.plan_decode)
                    for local, i in enumerate(batch_ids):
                        p = normalize_one_line(batch_out[local])
                        if p:
                            plans[i] = p
                            metas[i]["plan"] = logger.try_parse_json(p) or p

                    if batch_ids:
                        logger.log_first_in_batch(
                            stage=f"PLAN REPLAN {retry_round}",
                            batch_index=batch_index,
                            batch_size=len(batch_ids),
                            example_index=batch_ids[0],
                            chat_text=batch_texts[0],
                            model_output=batch_out[0] if batch_out else "",
                            meta=metas[batch_ids[0]],
                        )

            new_bad: List[int] = []
            for batch_index, batch_ids in enumerate(batched(bad_indices, self.cfg.batch_size), start=1):
                batch_texts: List[str] = []
                for i in batch_ids:
                    msgs = self._build_final_messages(df, i, plans[i])
                    batch_texts.append(to_chat_text(tokenizer, msgs))

                batch_out = generate_batch_once(model, tokenizer, batch_texts, decode)

                for local, i in enumerate(batch_ids):
                    out = normalize_one_line(batch_out[local])
                    if out:
                        outputs[i] = out
                        last_before_fallback[i] = out

                if batch_ids:
                    i0 = batch_ids[0]
                    w1, w2 = self._anchors_for_row(df, i0)
                    evaluation = None
                    try:
                        evaluation = self._eval_dict(outputs[i0], w1, w2)
                    except Exception:
                        evaluation = None

                    logger.log_first_in_batch(
                        stage=f"FINAL RETRY {retry_round}",
                        batch_index=batch_index,
                        batch_size=len(batch_ids),
                        example_index=i0,
                        chat_text=batch_texts[0],
                        model_output=batch_out[0] if batch_out else "",
                        meta=metas[i0],
                        evaluation=evaluation,
                        note="strict_suffix=true",
                    )

                for i in batch_ids:
                    if not self._is_good(df, i, outputs[i]):
                        new_bad.append(i)
                        ever_failed.add(i)

            bad_indices = new_bad

        fallback_items: List[Dict[str, Any]] = []
        if bad_indices:
            print(f"\nApplying fallback for remaining failures: {len(bad_indices)}")
            for i in bad_indices:
                used_fallback[i] = True
                outputs[i] = self._fallback_output(df, i)
                w1, w2 = self._anchors_for_row(df, i)
                fallback_items.append(
                    {
                        "index": int(i),
                        "anchors": {"word1": w1, "word2": w2},
                        "headline": metas[i].get("headline"),
                        "final_prediction": outputs[i],
                    }
                )

        # ---------------------------------------------------------------------
        # Save predictions
        # ---------------------------------------------------------------------
        out_rows: List[Tuple[str, str]] = []
        for i in range(n):
            out_id = str(df.loc[i, "id"]) if "id" in df.columns else str(i)
            out_rows.append((out_id, outputs[i]))
        out_df = pd.DataFrame(out_rows, columns=["id", "joke"])
        out_df.to_csv(out_path, sep="\t", index=False)

        zip_path = zip_file(out_path)

        # ---------------------------------------------------------------------
        # Log summaries at the end
        # ---------------------------------------------------------------------
        failed_items: List[Dict[str, Any]] = []
        for i in sorted(ever_failed):
            w1, w2 = self._anchors_for_row(df, i)
            eval_final = None
            try:
                evaluator = self._ensure_evaluator()
                eval_final = {
                    "required_words_present": bool(evaluator.required_words_present(outputs[i], w1, w2)),
                    "humorous": bool(evaluator.is_humorous(outputs[i])),
                    "good": bool(self._is_good(df, i, outputs[i])),
                }
            except Exception:
                eval_final = None

            failed_items.append(
                {
                    "index": int(i),
                    "anchors": {"word1": w1, "word2": w2},
                    "headline": metas[i].get("headline"),
                    "plan": metas[i].get("plan"),
                    "used_fallback": bool(used_fallback[i]),
                    "last_model_output_before_fallback": last_before_fallback[i],
                    "final_prediction": outputs[i],
                    "evaluation_final": eval_final,
                }
            )

        if failed_items:
            logger.log_failed_jokes(failed_items)

        if fallback_items:
            logger.log_fallback_predictions(fallback_items)

        plan_elapsed = float(time.time() - t_plan)
        final_elapsed = float(time.time() - t_final)
        logger.log_run_end(
            {
                "plan_seconds": round(plan_elapsed, 3),
                "final_seconds": round(final_elapsed, 3),
                "total_seconds": round(plan_elapsed + final_elapsed, 3),
                "rows": n,
                "ever_failed_count": len(ever_failed),
                "fallback_count": int(sum(1 for x in used_fallback if x)),
                "zip_path": str(zip_path),
                "log_path": str(log_path),
            }
        )
        logger.close()

        # Copy to Drive (zip + log)
        copy_outputs_to_drive([zip_path, log_path], self.cfg.drive_output_dir)

        print(f"\nSaving predictions to: {out_path}")
        print(f"Zipping to: {zip_path}")
        print("Done.")


# =============================================================================
# CLI
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print_config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_runner_config(Path(args.config))

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    runner = InferenceRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
