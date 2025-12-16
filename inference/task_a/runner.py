from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    do_sample: bool = True


@dataclass(frozen=True)
class RetryStep:
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class RunnerConfig:
    # "two_words" or "title"
    task: str

    base_model_id: str

    input_path: Path
    output_dir: Path
    output_filename: str

    batch_size: int = 16

    plan_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=72,
        min_new_tokens=16,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
    )

    final_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=64,
        min_new_tokens=10,
        temperature=0.9,
        top_p=0.98,
        do_sample=True,
    )

    max_retries: int = 5
    retry_steps: Tuple[RetryStep, ...] = (
        RetryStep(0.90, 0.95, 40),
        RetryStep(0.95, 0.98, 40),
        RetryStep(1.00, 0.98, 48),
        RetryStep(1.05, 0.98, 48),
        RetryStep(0.90, 0.95, 40),
    )

    # Title-only knobs
    noun_seed_base: int = 1337
    replan_every: int = 3  # 0 disables replanning

    # Colab drive copy
    drive_output_dir: str = "/content/drive/MyDrive/MWAHAHA_outputs"

    # Log file name. If empty, it becomes "<output_filename>.log.json"
    log_filename: str = ""


# =============================================================================
# Loading config from a python file
# =============================================================================

def load_runner_config(config_path: Path) -> RunnerConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("runner_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config: {config_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_config"):
        cfg = mod.get_config()
        if not isinstance(cfg, RunnerConfig):
            raise TypeError("get_config() must return RunnerConfig")
        return cfg

    if not hasattr(mod, "RUNNER_CONFIG"):
        raise AttributeError("Config file must define RUNNER_CONFIG or get_config().")

    obj = getattr(mod, "RUNNER_CONFIG")

    if isinstance(obj, RunnerConfig):
        return obj

    if isinstance(obj, dict):
        data = dict(obj)

        if "input_path" in data and isinstance(data["input_path"], str):
            data["input_path"] = Path(data["input_path"])
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])

        def _maybe_decode(key: str) -> None:
            if key in data and isinstance(data[key], dict):
                data[key] = DecodeConfig(**data[key])

        _maybe_decode("plan_decode")
        _maybe_decode("final_decode")

        if "retry_steps" in data and isinstance(data["retry_steps"], (list, tuple)):
            rs = []
            for x in data["retry_steps"]:
                rs.append(RetryStep(**x) if isinstance(x, dict) else x)
            data["retry_steps"] = tuple(rs)

        return RunnerConfig(**data)

    raise TypeError("RUNNER_CONFIG must be RunnerConfig or dict")


# =============================================================================
# Helpers
# =============================================================================

def batched(seq: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), batch_size):
        yield list(seq[i : i + batch_size])


def order_by_length(texts: List[str]) -> List[int]:
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


def to_chat_text(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def generate_batch_once(model, tokenizer, chat_texts: List[str], decode: DecodeConfig) -> List[str]:
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    prompt_len = input_ids.shape[1]

    bad_phrases = [
        "assistant", "assistant:", "user", "user:", "system", "system:",
        "assistant user", "<tool call>", "</tool call>",
    ]
    bad_words_ids = tokenizer(bad_phrases, add_special_tokens=False).input_ids

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=decode.max_new_tokens,
        min_new_tokens=decode.min_new_tokens,
        do_sample=decode.do_sample,
        temperature=decode.temperature,
        top_p=decode.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=bad_words_ids,
    )

    new_tokens = gen_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [normalize_one_line(x) for x in decoded]


def generate_all(
    model,
    tokenizer,
    chat_texts: List[str],
    batch_size: int,
    decode: DecodeConfig,
    label: str,
    *,
    logger: Optional[Logger] = None,
    metas: Optional[List[Dict[str, Any]]] = None,
    eval_fn: Optional[Callable[[str, Optional[Dict[str, Any]]], Optional[Dict[str, Any]]]] = None,
    note: Optional[str] = None,
) -> List[str]:
    n = len(chat_texts)
    outputs: List[str] = [""] * n
    order = order_by_length(chat_texts)

    for batch_index, batch_ids in enumerate(batched(order, batch_size), start=1):
        batch_texts = [chat_texts[i] for i in batch_ids]

        # stdout: progress only
        print(f"\n[{label} batch {batch_index}] size={len(batch_ids)}")

        batch_out = generate_batch_once(model, tokenizer, batch_texts, decode)

        # file: structured extra logs only
        if logger is not None and batch_ids:
            first_local = 0
            first_global = batch_ids[first_local]

            meta = None
            if metas is not None and 0 <= first_global < len(metas):
                meta = metas[first_global]

            evaluation = None
            if eval_fn is not None:
                try:
                    evaluation = eval_fn(batch_out[first_local], meta)
                except Exception as exc:
                    evaluation = {"error": f"{type(exc).__name__}: {exc}"}

            logger.log_first_in_batch(
                stage=label,
                batch_index=batch_index,
                batch_size=len(batch_ids),
                example_index=first_global,
                chat_text=batch_texts[first_local],
                model_output=batch_out[first_local],
                meta=meta,
                evaluation=evaluation,
                note=note,
            )

        for j, out_text in enumerate(batch_out):
            outputs[batch_ids[j]] = out_text

    return outputs


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


def copy_files_to_drive(paths: List[Path], drive_dir: str) -> None:
    # Works only in Colab with Drive mounted. Safe no-op elsewhere.
    try:
        from google.colab import drive as colab_drive  # noqa: F401
    except Exception:
        print("Not running in Colab, skipping Google Drive upload.")
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("Google Drive is not mounted.")
        return

    drive_dir_path = Path(drive_dir)
    if str(drive_dir_path).startswith("/content/drive"):
        dest_dir = drive_root / drive_dir_path.relative_to("/content/drive")
    else:
        dest_dir = drive_root / drive_dir_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    for p in paths:
        if not p.exists():
            continue
        dest_path = dest_dir / p.name
        shutil.copy2(p, dest_path)
        print("Copied to Google Drive:", dest_path)


# =============================================================================
# Runner
# =============================================================================

@dataclass
class InferenceRunner:
    cfg: RunnerConfig
    builder: PromptBuilder = field(default_factory=PromptBuilder)
    evaluator: ResponseEvaluator = field(default_factory=ResponseEvaluator)

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
            df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
            if "headline" not in df.columns:
                raise ValueError(f"Missing required column 'headline' in {self.cfg.input_path}")
            df["headline"] = df["headline"].fillna("").astype(str)

            noun1_list: List[str] = []
            noun2_list: List[str] = []
            for i, row in df.iterrows():
                h = normalize_one_line(row["headline"])
                n1, n2 = self.builder.choose_two_nouns_from_headline(
                    h, seed=self.cfg.noun_seed_base + int(i), prefer_distinct=True
                )
                noun1_list.append(n1 or "news")
                noun2_list.append(n2 or "headline")

            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df

        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _fallback_plan(self) -> str:
        return normalize_one_line(
            '{"scenario":"everyday misunderstanding","misdirection":"literal vs figurative","anchor_placement":"use both anchors in punchline","device":"reversal"}'
        )

    def _fallback_output_two_words(self, w1: str, w2: str) -> str:
        w1 = safe_word(w1) or "thing"
        w2 = safe_word(w2) or "place"
        return normalize_one_line(f"I brought a {w1} to a {w2} meeting. Now I am somehow in charge of both.")

    def _fallback_output_title(self, headline: str, n1: str, n2: str) -> str:
        headline = normalize_one_line(headline)
        n1 = safe_word(n1) or "news"
        n2 = safe_word(n2) or "headline"
        if not headline:
            return normalize_one_line(f"My {n1} met a {n2}. Somehow it still became breaking news.")
        return normalize_one_line(f"After '{headline}', my {n1} hired a {n2} for public relations. Big mistake.")

    def _is_good(self, text: str, a1: str, a2: str) -> bool:
        text = normalize_one_line(text)
        if not text:
            return False
        return self.evaluator.is_good(text, a1, a2)

    def run(self) -> Path:
        df = self._load_dataframe()
        n = len(df)

        model, tokenizer = load_model_and_tokenizer(self.cfg.base_model_id)

        # Log file path in output directory
        log_name = self.cfg.log_filename.strip()
        if not log_name:
            log_name = f"{self.cfg.output_filename}.log.json"
        log_path = self.cfg.output_dir / log_name

        logger = Logger(enabled=True, log_path=log_path, max_chars=900)

        start_all = time.time()
        ever_failed: Set[int] = set()
        fallback_used: Set[int] = set()

        try:
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

            # Per-row metadata for logging
            metas: List[Dict[str, Any]] = []
            for i in range(n):
                if self.cfg.task == "two_words":
                    metas.append(
                        {
                            "anchors": {"word1": str(df.loc[i, "word1"]), "word2": str(df.loc[i, "word2"])},
                            "headline": None,
                            "wikipedia_context": "",
                            "microcards": [],
                            "plan": None,
                        }
                    )
                else:
                    metas.append(
                        {
                            "anchors": {"noun1": str(df.loc[i, "noun1"]), "noun2": str(df.loc[i, "noun2"])},
                            "headline": str(df.loc[i, "headline"]),
                            "wikipedia_context": "",
                            "microcards": [],
                            "plan": None,
                        }
                    )

            # -------------------------
            # PLAN stage
            # -------------------------
            if self.cfg.task == "two_words":
                plan_texts = [
                    to_chat_text(
                        tokenizer,
                        self.builder.build_two_words_plan_messages(df.loc[i, "word1"], df.loc[i, "word2"]),
                    )
                    for i in range(n)
                ]
            else:
                plan_texts = [
                    to_chat_text(
                        tokenizer,
                        self.builder.build_title_plan_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"]),
                    )
                    for i in range(n)
                ]

            # Extract Wikipedia context and microcards from prompts (best-effort)
            for i, chat_text in enumerate(plan_texts):
                blocks = logger.extract_prompt_blocks(chat_text)
                metas[i]["wikipedia_context"] = blocks.get("facts", "") or ""
                metas[i]["microcards"] = blocks.get("microcards", []) or []

            plans = generate_all(
                model, tokenizer, plan_texts, self.cfg.batch_size, self.cfg.plan_decode, "PLAN",
                logger=logger, metas=metas
            )

            # Fill empty plans
            for i, p in enumerate(plans):
                if not normalize_one_line(p):
                    plans[i] = self._fallback_plan()
                metas[i]["plan"] = logger.try_parse_json(plans[i]) or plans[i]

            # -------------------------
            # FINAL stage
            # -------------------------
            if self.cfg.task == "two_words":
                final_texts = [
                    to_chat_text(
                        tokenizer,
                        self.builder.build_two_words_final_messages(df.loc[i, "word1"], df.loc[i, "word2"], plans[i]),
                    )
                    for i in range(n)
                ]
            else:
                final_texts = [
                    to_chat_text(
                        tokenizer,
                        self.builder.build_title_final_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"], plans[i]),
                    )
                    for i in range(n)
                ]

            def eval_fn_for_joke(output_text: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
                anchors = (meta or {}).get("anchors") or {}
                a1 = anchors.get("word1") or anchors.get("noun1") or ""
                a2 = anchors.get("word2") or anchors.get("noun2") or ""
                required_ok = self.evaluator.required_words_present(output_text, a1, a2)
                humorous_ok = self.evaluator.is_humorous(output_text)
                return {
                    "required_words": {"anchor1": a1, "anchor2": a2, "present": bool(required_ok)},
                    "humor_classifier": {"humorous": bool(humorous_ok)},
                    "overall": {"good": bool(required_ok and humorous_ok)},
                }

            outputs = generate_all(
                model, tokenizer, final_texts, self.cfg.batch_size, self.cfg.final_decode, "FINAL",
                logger=logger, metas=metas, eval_fn=eval_fn_for_joke
            )

            # Identify failures after initial final
            bad_indices: List[int] = []
            for i in range(n):
                if self.cfg.task == "two_words":
                    if not self._is_good(outputs[i], df.loc[i, "word1"], df.loc[i, "word2"]):
                        bad_indices.append(i)
                else:
                    if not self._is_good(outputs[i], df.loc[i, "noun1"], df.loc[i, "noun2"]):
                        bad_indices.append(i)

            ever_failed.update(bad_indices)
            print(f"\nInitial final failures: {len(bad_indices)}")

            strict_suffix = "IMPORTANT: The answer is INVALID unless it uses both required anchors. Output only the joke."

            # -------------------------
            # RETRY loop
            # -------------------------
            for retry_round, step in enumerate(self.cfg.retry_steps[: self.cfg.max_retries], start=1):
                if not bad_indices:
                    break

                retry_decode = DecodeConfig(
                    max_new_tokens=step.max_new_tokens,
                    min_new_tokens=max(self.cfg.final_decode.min_new_tokens, 8),
                    temperature=step.temperature,
                    top_p=step.top_p,
                    do_sample=self.cfg.final_decode.do_sample,
                )

                retry_texts: List[str] = []
                for i in bad_indices:
                    if self.cfg.task == "two_words":
                        msgs = self.builder.build_two_words_final_messages(df.loc[i, "word1"], df.loc[i, "word2"], plans[i])
                    else:
                        msgs = self.builder.build_title_final_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"], plans[i])

                    msgs = [dict(msgs[0]), dict(msgs[1])]
                    msgs[1]["content"] = normalize_one_line((msgs[1]["content"] or "") + "\n" + strict_suffix)
                    retry_texts.append(to_chat_text(tokenizer, msgs))

                retry_metas: List[Dict[str, Any]] = [metas[i] for i in bad_indices]

                retry_out = generate_all(
                    model,
                    tokenizer,
                    retry_texts,
                    self.cfg.batch_size,
                    retry_decode,
                    f"FINAL RETRY {retry_round}",
                    logger=logger,
                    metas=retry_metas,
                    eval_fn=eval_fn_for_joke,
                    note="strict_suffix=true",
                )

                new_bad: List[int] = []
                for pos, i in enumerate(bad_indices):
                    candidate = normalize_one_line(retry_out[pos])
                    if candidate:
                        outputs[i] = candidate

                    if self.cfg.task == "two_words":
                        ok = self._is_good(outputs[i], df.loc[i, "word1"], df.loc[i, "word2"])
                    else:
                        ok = self._is_good(outputs[i], df.loc[i, "noun1"], df.loc[i, "noun2"])

                    if not ok:
                        new_bad.append(i)

                bad_indices = new_bad
                ever_failed.update(bad_indices)
                print(f"After retry {retry_round}: remaining failures = {len(bad_indices)}")

                # Optional replanning for title task
                if (
                    bad_indices
                    and self.cfg.task == "title"
                    and self.cfg.replan_every > 0
                    and (retry_round % self.cfg.replan_every == 0)
                ):
                    replan_texts = [
                        to_chat_text(
                            tokenizer,
                            self.builder.build_title_plan_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"]),
                        )
                        for i in bad_indices
                    ]
                    replan_metas: List[Dict[str, Any]] = [metas[i] for i in bad_indices]

                    replan_out = generate_all(
                        model,
                        tokenizer,
                        replan_texts,
                        self.cfg.batch_size,
                        self.cfg.plan_decode,
                        f"REPLAN {retry_round}",
                        logger=logger,
                        metas=replan_metas,
                    )

                    for pos, i in enumerate(bad_indices):
                        p = normalize_one_line(replan_out[pos])
                        if p:
                            plans[i] = p
                            metas[i]["plan"] = logger.try_parse_json(p) or p

            # -------------------------
            # Fallback for remaining failures
            # -------------------------
            pre_fallback_outputs = outputs[:]  # keep for log
            if bad_indices:
                print(f"\nApplying fallback for remaining failures: {len(bad_indices)}")
                for i in bad_indices:
                    fallback_used.add(i)
                    if self.cfg.task == "two_words":
                        outputs[i] = self._fallback_output_two_words(df.loc[i, "word1"], df.loc[i, "word2"])
                    else:
                        outputs[i] = self._fallback_output_title(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"])

            # Final validation (must pass)
            still_bad: List[int] = []
            for i in range(n):
                if self.cfg.task == "two_words":
                    if not self._is_good(outputs[i], df.loc[i, "word1"], df.loc[i, "word2"]):
                        still_bad.append(i)
                else:
                    if not self._is_good(outputs[i], df.loc[i, "noun1"], df.loc[i, "noun2"]):
                        still_bad.append(i)
            if still_bad:
                raise RuntimeError(f"Validation failed for indices: {still_bad[:20]}")

            # Save
            df_out = df.copy()
            df_out["prediction"] = outputs
            zip_path = save_and_zip(df_out, self.cfg.output_dir, self.cfg.output_filename)

            # Append failed jokes list to log file (ALL indices that ever failed)
            failed_items: List[Dict[str, Any]] = []
            for i in sorted(ever_failed):
                meta = metas[i]
                anchors = meta.get("anchors") or {}

                a1 = anchors.get("word1") or anchors.get("noun1") or ""
                a2 = anchors.get("word2") or anchors.get("noun2") or ""

                # evaluation can be expensive, but this is only for failed set
                try:
                    eval_final = {
                        "required_words_present": bool(self.evaluator.required_words_present(outputs[i], a1, a2)),
                        "humorous": bool(self.evaluator.is_humorous(outputs[i])),
                    }
                    eval_final["good"] = bool(eval_final["required_words_present"] and eval_final["humorous"])
                except Exception as exc:
                    eval_final = {"error": f"{type(exc).__name__}: {exc}"}

                failed_items.append(
                    {
                        "index": i,
                        "anchors": anchors,
                        "headline": meta.get("headline"),
                        "plan": meta.get("plan"),
                        "used_fallback": i in fallback_used,
                        "last_model_output_before_fallback": pre_fallback_outputs[i] if i < len(pre_fallback_outputs) else "",
                        "final_prediction": outputs[i],
                        "evaluation_final": eval_final,
                    }
                )

            logger.log_failed_jokes(failed_items)

            elapsed = time.time() - start_all
            logger.log_run_end(
                {
                    "total_seconds": round(elapsed, 3),
                    "rows": n,
                    "ever_failed_count": len(ever_failed),
                    "fallback_count": len(fallback_used),
                    "zip_path": str(zip_path),
                    "log_path": str(log_path),
                }
            )

            # Copy zip + log to Drive
            copy_files_to_drive([zip_path, log_path], self.cfg.drive_output_dir)

            print(f"\nDone. Total seconds: {elapsed:.1f}")
            return zip_path

        finally:
            logger.close()


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config.py defining RUNNER_CONFIG or get_config()")
    p.add_argument("--print_config", action="store_true", help="Print loaded config and exit")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_runner_config(Path(args.config))

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    runner = InferenceRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
