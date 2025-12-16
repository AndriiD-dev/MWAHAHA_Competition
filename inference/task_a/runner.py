from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
    task: str  # "two_words" or "title"

    base_model_id: str
    input_path: Path
    output_dir: Path
    output_filename: str

    batch_size: int = 8

    plan_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=96,
        min_new_tokens=16,
        temperature=0.7,
        top_p=0.95,
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

    # Google Drive (optional)
    drive_output_dir: Optional[str] = None


# =============================================================================
# Loading config from a python file
# =============================================================================

def load_runner_config(config_path: Path) -> RunnerConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("runner_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "RUNNER_CONFIG"):
        raise AttributeError("Config file must define RUNNER_CONFIG")

    rc = getattr(module, "RUNNER_CONFIG")

    if isinstance(rc, RunnerConfig):
        return rc
    if isinstance(rc, dict):
        # Allow paths as strings
        if "input_path" in rc and isinstance(rc["input_path"], str):
            rc["input_path"] = Path(rc["input_path"])
        if "output_dir" in rc and isinstance(rc["output_dir"], str):
            rc["output_dir"] = Path(rc["output_dir"])

        def _maybe_decode(key: str) -> None:
            if key in rc and isinstance(rc[key], dict):
                rc[key] = DecodeConfig(**rc[key])

        _maybe_decode("plan_decode")
        _maybe_decode("final_decode")

        if "retry_steps" in rc and isinstance(rc["retry_steps"], (list, tuple)):
            rs = []
            for x in rc["retry_steps"]:
                rs.append(RetryStep(**x) if isinstance(x, dict) else x)
            rc["retry_steps"] = tuple(rs)

        return RunnerConfig(**rc)

    raise TypeError("RUNNER_CONFIG must be RunnerConfig or dict")


# =============================================================================
# Small helpers
# =============================================================================

def configure_fast_kernels() -> None:
    # Safe defaults; do not crash if unsupported.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=decode.max_new_tokens,
            min_new_tokens=decode.min_new_tokens,
            temperature=decode.temperature,
            top_p=decode.top_p,
            do_sample=decode.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
    return decoded


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
        print(f"\n[{label} batch {batch_index}] size={len(batch_ids)}")

        batch_out = generate_batch_once(model, tokenizer, batch_texts, decode)

        # Log only the first sample in the batch.
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


def copy_to_drive(zip_path: Path, drive_dir: Optional[str]) -> None:
    if not drive_dir:
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("Drive not mounted; skipping copy_to_drive.")
        return

    dest_dir = drive_root / Path(drive_dir).relative_to("/content/drive")
    dest_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    dest_path = dest_dir / zip_path.name
    shutil.copy2(zip_path, dest_path)
    print("Copied zip to Google Drive:", dest_path)


# =============================================================================
# Runner class (handles both tasks)
# =============================================================================

@dataclass
class InferenceRunner:
    cfg: RunnerConfig
    builder: PromptBuilder = field(default_factory=PromptBuilder)
    response_evaluator: ResponseEvaluator = field(default_factory=ResponseEvaluator)

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
                if h == "":
                    noun1_list.append("news")
                    noun2_list.append("headline")
                    continue

                n1, n2 = self.builder.pick_nouns_from_headline(h, seed=self.cfg.noun_seed_base + int(i))
                noun1_list.append(n1 or "news")
                noun2_list.append(n2 or "headline")

            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df

        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _is_good(self, df: pd.DataFrame, i: int, out: str) -> bool:
        out = normalize_one_line(out)
        if out == "":
            return False

        if self.cfg.task == "two_words":
            return self.response_evaluator.is_good(out, df.loc[i, "word1"], df.loc[i, "word2"])

        return self.response_evaluator.is_good(out, df.loc[i, "noun1"], df.loc[i, "noun2"])

    def _fallback_plan(self, df: pd.DataFrame, i: int) -> str:
        _ = (df, i)
        if self.cfg.task == "two_words":
            return normalize_one_line(
                '{"scenario":"everyday misunderstanding","misdirection":"use word1 as a verb, then reveal it is a noun","anchor_placement":"use both words in the punchline","device":"reversal"}'
            )
        return normalize_one_line(
            '{"scenario":"news headline twist","misdirection":"literal reading of the headline","anchor_placement":"use both nouns in the punchline","device":"escalation"}'
        )

    def _fallback_output(self, df: pd.DataFrame, i: int) -> str:
        if self.cfg.task == "two_words":
            w1 = safe_word(df.loc[i, "word1"])
            w2 = safe_word(df.loc[i, "word2"])
            return normalize_one_line(f"I brought a {w1} to a {w2} meeting; both were unqualified, but somehow still got promoted.")

        h = normalize_one_line(df.loc[i, "headline"])
        n1 = safe_word(df.loc[i, "noun1"]) or "news"
        n2 = safe_word(df.loc[i, "noun2"]) or "headline"
        if not h:
            return normalize_one_line(f"My {n1} met a {n2}; somehow both still made the evening news.")
        return normalize_one_line(
            f"After '{h}', my {n1} hired a {n2} for public relations. It went exactly as well as you think."
        )

    def run(self) -> Path:
        configure_fast_kernels()

        print("Project root:", self.builder.config.paths.project_root)
        print("Using input :", self.cfg.input_path)
        print("Using output:", self.cfg.output_dir / self.cfg.output_filename)

        model, tokenizer = load_model_and_tokenizer(self.cfg.base_model_id)

        df = self._load_dataframe()
        n = len(df)
        start_all = time.time()

        logger = Logger(enabled=True, max_chars=900)

        metas: List[Dict[str, Any]] = []
        for i in range(n):
            if self.cfg.task == "two_words":
                metas.append(
                    {
                        "anchors": {
                            "word1": str(df.loc[i, "word1"]),
                            "word2": str(df.loc[i, "word2"]),
                        },
                        "headline": None,
                        "wikipedia_context": "",
                        "microcards": [],
                        "plan": None,
                    }
                )
            else:
                metas.append(
                    {
                        "anchors": {
                            "noun1": str(df.loc[i, "noun1"]),
                            "noun2": str(df.loc[i, "noun2"]),
                        },
                        "headline": str(df.loc[i, "headline"]),
                        "wikipedia_context": "",
                        "microcards": [],
                        "plan": None,
                    }
                )

        # Pass 1: plans
        if self.cfg.task == "two_words":
            plan_texts = [
                to_chat_text(tokenizer, self.builder.build_two_words_plan_messages(df.loc[i, "word1"], df.loc[i, "word2"]))
                for i in range(n)
            ]
        else:
            plan_texts = [
                to_chat_text(tokenizer, self.builder.build_title_plan_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"]))
                for i in range(n)
            ]

        # Extract Wikipedia context and microcards from the prompt text (best-effort).
        for i, chat_text in enumerate(plan_texts):
            blocks = logger.extract_prompt_blocks(chat_text)
            metas[i]["wikipedia_context"] = blocks.get("facts", "") or ""
            metas[i]["microcards"] = blocks.get("microcards", []) or []

        plans = generate_all(model, tokenizer, plan_texts, self.cfg.batch_size, self.cfg.plan_decode, "PLAN", logger=logger, metas=metas)
        for i, p in enumerate(plans):
            if normalize_one_line(p) == "":
                plans[i] = self._fallback_plan(df, i)

        for i, p in enumerate(plans):
            metas[i]["plan"] = logger.try_parse_json(p) or p

        # Pass 2: finals
        if self.cfg.task == "two_words":
            final_texts = [
                to_chat_text(tokenizer, self.builder.build_two_words_final_messages(df.loc[i, "word1"], df.loc[i, "word2"], plans[i]))
                for i in range(n)
            ]
        else:
            final_texts = [
                to_chat_text(tokenizer, self.builder.build_title_final_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"], plans[i]))
                for i in range(n)
            ]

        def eval_fn_for_joke(output_text: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            anchors = (meta or {}).get("anchors") or {}
            word1 = anchors.get("word1") or anchors.get("noun1") or ""
            word2 = anchors.get("word2") or anchors.get("noun2") or ""

            required_ok = self.response_evaluator.required_words_present(output_text, word1, word2)
            humorous_ok = self.response_evaluator.is_humorous(output_text)
            good_ok = bool(required_ok and humorous_ok)

            return {
                "required_words": {
                    "word1": word1,
                    "word2": word2,
                    "present": bool(required_ok),
                },
                "humor_classifier": {
                    "humorous": bool(humorous_ok),
                },
                "overall": {
                    "good": good_ok,
                },
            }

        outputs = generate_all(model, tokenizer, final_texts, self.cfg.batch_size, self.cfg.final_decode, "FINAL", logger=logger, metas=metas, eval_fn=eval_fn_for_joke)

        bad_indices = [i for i in range(n) if not self._is_good(df, i, outputs[i])]
        print(f"\nInitial final failures: {len(bad_indices)}")

        strict_suffix = (
            "IMPORTANT: The answer is INVALID unless it satisfies the required anchor constraint. Output only the joke."
        )

        # Retry final for failed indices
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
                msgs[1]["content"] = normalize_one_line(msgs[1]["content"] + "\n" + strict_suffix)
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
                if candidate and self._is_good(df, i, candidate):
                    outputs[i] = candidate
                else:
                    new_bad.append(i)
            bad_indices = new_bad
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
                    plans[i] = p if p else plans[i]
                    if p:
                        metas[i]["plan"] = logger.try_parse_json(p) or p

        # Fallback for remaining failures
        if bad_indices:
            print(f"\nApplying fallback for remaining failures: {len(bad_indices)}")
            for i in bad_indices:
                outputs[i] = self._fallback_output(df, i)

        still_bad = [i for i in range(n) if not self._is_good(df, i, outputs[i])]
        if still_bad:
            raise RuntimeError(f"Validation failed for indices: {still_bad[:20]}")

        df_out = df.copy()
        df_out["prediction"] = outputs

        zip_path = save_and_zip(df_out, self.cfg.output_dir, self.cfg.output_filename)
        copy_to_drive(zip_path, self.cfg.drive_output_dir)

        elapsed = time.time() - start_all
        print(f"\nDone. Total seconds: {elapsed:.1f}")
        return zip_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to config.py defining RUNNER_CONFIG")
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
