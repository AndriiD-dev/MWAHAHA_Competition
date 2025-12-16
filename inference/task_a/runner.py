from __future__ import annotations

import argparse
import importlib.util
import json
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.prompt_builder import PromptBuilder, normalize_one_line, safe_word


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
        max_new_tokens=32,
        min_new_tokens=6,
        temperature=0.8,
        top_p=0.95,
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


# =============================================================================
# Config loader (Python file path)
# =============================================================================

def load_runner_config(config_path: Path) -> RunnerConfig:
    """
    Config file must define:
        RUNNER_CONFIG = RunnerConfig(...)
    or:
        RUNNER_CONFIG = { ... } (keys match RunnerConfig fields)
    or:
        def get_config() -> RunnerConfig: ...
    """
    config_path = config_path.resolve()
    spec = importlib.util.spec_from_file_location("mwahaha_run_config", str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

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

        # Allow string paths
        for k in ("input_path", "output_dir"):
            if k in data and isinstance(data[k], str):
                data[k] = Path(data[k])

        # Nested dataclasses from dict
        if "plan_decode" in data and isinstance(data["plan_decode"], dict):
            data["plan_decode"] = DecodeConfig(**data["plan_decode"])
        if "final_decode" in data and isinstance(data["final_decode"], dict):
            data["final_decode"] = DecodeConfig(**data["final_decode"])
        if "retry_steps" in data and isinstance(data["retry_steps"], (list, tuple)):
            steps: List[RetryStep] = []
            for it in data["retry_steps"]:
                if isinstance(it, RetryStep):
                    steps.append(it)
                elif isinstance(it, dict):
                    steps.append(RetryStep(**it))
                else:
                    raise TypeError("retry_steps items must be RetryStep or dict")
            data["retry_steps"] = tuple(steps)

        return RunnerConfig(**data)

    raise TypeError("RUNNER_CONFIG must be RunnerConfig or dict")


# =============================================================================
# Utilities
# =============================================================================

def configure_fast_kernels() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def load_model_and_tokenizer(base_model_id: str):
    print("Loading tokenizer from base model:", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

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

    last_err = None
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            print(f"Loading base model with attention implementation: {attn_impl}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
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


def batched(items: List[int], batch_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def order_by_length(texts: List[str]) -> List[int]:
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


def to_chat_text(tokenizer, messages: List[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
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
) -> List[str]:
    n = len(chat_texts)
    outputs: List[str] = [""] * n
    order = order_by_length(chat_texts)

    for batch_index, batch_ids in enumerate(batched(order, batch_size), start=1):
        batch_texts = [chat_texts[i] for i in batch_ids]
        print(f"\n[{label} batch {batch_index}] size={len(batch_ids)}")
        batch_out = generate_batch_once(model, tokenizer, batch_texts, decode)
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


def copy_to_drive(zip_path: Path, drive_dir: str) -> None:
    try:
        from google.colab import drive as colab_drive  # noqa: F401
    except Exception:
        print("Not running in Colab, skipping Google Drive upload.")
        return

    drive_root = Path("/content/drive")
    if not drive_root.exists():
        print("Google Drive is not mounted.")
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
            for i, h in enumerate(df["headline"].tolist()):
                seed = int(self.cfg.noun_seed_base) + i
                n1, n2 = self.builder.choose_two_nouns_from_headline(h, seed=seed, prefer_distinct=True)
                noun1_list.append(n1)
                noun2_list.append(n2)

            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df

        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _is_good(self, df: pd.DataFrame, i: int, out: str) -> bool:
        out = normalize_one_line(out)
        if out == "":
            return False

        if self.cfg.task == "two_words":
            return self.builder.required_words_present(out, df.loc[i, "word1"], df.loc[i, "word2"])

        return self.builder.required_words_present(out, df.loc[i, "noun1"], df.loc[i, "noun2"])

    def _fallback_plan(self, df: pd.DataFrame, i: int) -> str:
        _ = (df, i)
        if self.cfg.task == "two_words":
            return normalize_one_line(
                '{"scenario":"everyday misunderstanding","misdirection":"take it literally","word_placement":"use both words in the punchline","device":"reversal"}'
            )
        return normalize_one_line(
            '{"angle":"obvious headline irony","misdirection":"literal reading","word_placement":"use both nouns in the punchline","device":"reversal"}'
        )

    def _fallback_output(self, df: pd.DataFrame, i: int) -> str:
        if self.cfg.task == "two_words":
            w1 = safe_word(df.loc[i, "word1"])
            w2 = safe_word(df.loc[i, "word2"])
            return normalize_one_line(
                f"I brought a {w1} to a {w2} meeting; both were unqualified, but somehow still got promoted."
            )

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

        plans = generate_all(model, tokenizer, plan_texts, self.cfg.batch_size, self.cfg.plan_decode, "PLAN")
        for i, p in enumerate(plans):
            if normalize_one_line(p) == "":
                plans[i] = self._fallback_plan(df, i)

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

        outputs = generate_all(model, tokenizer, final_texts, self.cfg.batch_size, self.cfg.final_decode, "FINAL")

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

            retry_out = generate_all(model, tokenizer, retry_texts, self.cfg.batch_size, retry_decode, f"FINAL RETRY {retry_round}")

            new_bad: List[int] = []
            for pos, i in enumerate(bad_indices):
                candidate = normalize_one_line(retry_out[pos])
                if candidate and self._is_good(df, i, candidate):
                    outputs[i] = candidate
                else:
                    new_bad.append(i)
            bad_indices = new_bad
            print(f"Final failures after retry round {retry_round}: {len(bad_indices)}")

            # Optional replanning for title
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
                replan_out = generate_all(model, tokenizer, replan_texts, self.cfg.batch_size, self.cfg.plan_decode, f"REPLAN {retry_round}")
                for pos, i in enumerate(bad_indices):
                    p = normalize_one_line(replan_out[pos])
                    plans[i] = p if p else plans[i]

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

        self.builder.save_wiki_cache()

        print(f"\nAll generations done in {time.time() - start_all:.1f}s total.")
        print("All done.")
        return zip_path


# =============================================================================
# Command line entry point
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MWAHAHA unified inference runner")
    parser.add_argument("--config", required=True, type=str, help="Path to a Python config file")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args(argv)

    cfg = load_runner_config(Path(args.config))

    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    runner = InferenceRunner(cfg=cfg)
    runner.run()


if __name__ == "__main__":
    main()
