from __future__ import annotations

import os
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
from peft import PeftModel

from inference.utils.prompt_builder import PromptBuilder, normalize_one_line, safe_word
from inference.utils.response_evaluator import ResponseEvaluator
from inference.utils.logger import Logger

# NEW (caption_mm)
from inference.utils.gif_frames import download_gif, extract_frames, GifFrameSettings
from inference.utils.vl_scene_extractor import VLSceneExtractor


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
    # "two_words", "title", or "caption_mm"
    task: str

    base_model_id: str
    input_path: Path
    output_dir: Path
    output_filename: str

    batch_size: int = 16

    # PLAN stage decoding
    plan_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=80,
        min_new_tokens=16,
        temperature=0.4,
        top_p=0.9,
    )

    # FINAL stage decoding
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
    replan_every: int = 3

    # Caption_mm knobs
    candidates_per_row: int = 4
    caption_max_words: int = 20
    gif_frames: GifFrameSettings = GifFrameSettings()

    # Optional: copy outputs to Google Drive
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

    task = str(d["task"])
    base_model_id = str(d["base_model_id"])
    input_path = _as_path(d["input_path"])
    output_dir = _as_path(d["output_dir"])
    output_filename = str(d["output_filename"])

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

    # caption_mm extras
    candidates_per_row = int(d.get("candidates_per_row", 4))
    caption_max_words = int(d.get("caption_max_words", 20))

    gif_frames_cfg = d.get("gif_frames") or {}
    gif_frames = GifFrameSettings(
        max_frames=int(gif_frames_cfg.get("max_frames", 4)),
        max_side=int(gif_frames_cfg.get("max_side", 512)),
        timeout_seconds=float(gif_frames_cfg.get("timeout_seconds", 60.0)),
    )

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
        candidates_per_row=candidates_per_row,
        caption_max_words=caption_max_words,
        gif_frames=gif_frames,
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
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_id: str, lora_adapter_path: str | None = None):
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

    # Apply LoRA adapter (adapter-only folder or HF repo id)
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        # merge for faster inference (optional)
        try:
            model = model.merge_and_unload()
        except Exception:
            pass

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
# Runner class
# =============================================================================

@dataclass
class InferenceRunner:
    cfg: RunnerConfig
    builder: PromptBuilder = field(default_factory=PromptBuilder)

    response_evaluator: Optional[ResponseEvaluator] = None

    def _ensure_evaluator(self) -> ResponseEvaluator:
        if self.response_evaluator is None:
            print("Initializing ResponseEvaluator (required-words checker + humor classifier)...")
            self.response_evaluator = ResponseEvaluator()
        return self.response_evaluator


    def _load_dataframe(self) -> pd.DataFrame:
        if self.cfg.task == "two_words":
            df = pd.read_csv(self.cfg.input_path, sep="	", keep_default_na=False)
            need = {"word1", "word2"}
            missing = need.difference(set(df.columns))
            if missing:
                raise ValueError(f"Missing required columns in {self.cfg.input_path}: {sorted(missing)}")
            df["word1"] = df["word1"].fillna("").astype(str)
            df["word2"] = df["word2"].fillna("").astype(str)
            return df

        if self.cfg.task == "title":
            # Title files are typically TSV even if named *.csv
            try:
                df = pd.read_csv(self.cfg.input_path, sep="	", keep_default_na=False)
            except Exception:
                lines = Path(self.cfg.input_path).read_text(encoding="utf-8", errors="replace").splitlines()
                if lines and lines[0].strip().lower() == "headline":
                    lines = lines[1:]
                df = pd.DataFrame({"headline": lines})

            if "headline" not in df.columns:
                if len(df.columns) == 1:
                    df = df.rename(columns={df.columns[0]: "headline"})
                elif len(df.columns) >= 2:
                    df = df.rename(columns={df.columns[1]: "headline"})

            if "headline" not in df.columns:
                raise ValueError(f"Missing required column 'headline' in {self.cfg.input_path}")

            df["headline"] = df["headline"].fillna("").astype(str)

            print("Preparing nouns for each headline (spaCy noun extraction)...")

            noun1_list: List[str] = []
            noun2_list: List[str] = []
            n = len(df)

            t0 = time.time()
            for i in range(n):
                headline = df.loc[i, "headline"]
                seed = int(self.cfg.noun_seed_base + i)
                n1, n2 = self.builder.choose_two_nouns_from_headline(headline, seed=seed)
                noun1_list.append(n1)
                noun2_list.append(n2)

                if (i + 1) % 50 == 0 or (i + 1) == n:
                    dt = time.time() - t0
                    print(f"  nouns: {i + 1}/{n}  (elapsed {dt:.1f}s)")

            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df

        if self.cfg.task == "caption_mm":
            # Expected TSV like task-b1: columns [id, url] (url is a GIF link)
            df = pd.read_csv(self.cfg.input_path, sep="	", keep_default_na=False)

            if "url" in df.columns and "gif_url" not in df.columns:
                df["gif_url"] = df["url"]

            if "gif_url" not in df.columns and "gif_path" not in df.columns:
                raise ValueError("caption_mm requires column gif_url (or url) or gif_path")

            if "id" not in df.columns:
                # Create stable IDs if missing
                df.insert(0, "id", [f"row_{i:05d}" for i in range(len(df))])

            # Normalize to strings
            if "gif_url" in df.columns:
                df["gif_url"] = df["gif_url"].fillna("").astype(str)
            if "gif_path" in df.columns:
                df["gif_path"] = df["gif_path"].fillna("").astype(str)
            df["id"] = df["id"].fillna("").astype(str)

            return df

        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _anchors_for_row(self, df: pd.DataFrame, i: int) -> Tuple[str, str]:
        if self.cfg.task == "two_words":
            return str(df.loc[i, "word1"]), str(df.loc[i, "word2"])
        if self.cfg.task == "title":
            return str(df.loc[i, "noun1"]), str(df.loc[i, "noun2"])
        if self.cfg.task == "caption_mm":
            return str(df.loc[i, "noun1"]), str(df.loc[i, "noun2"])
        return ("", "")

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

        if self.cfg.task == "caption_mm":
            # Keep it short, includes both words; internal max-word validator will still apply.
            return normalize_one_line(f"My {w1} met my {w2} and suddenly the universe started heckling me.")

        return normalize_one_line(f"I tried to {w1} my life together, but my {w2} had other plans.")

    def _build_plan_messages(self, df: pd.DataFrame, i: int) -> List[Dict[str, str]]:
        if self.cfg.task == "two_words":
            return self.builder.build_two_words_plan_messages(df.loc[i, "word1"], df.loc[i, "word2"])
        if self.cfg.task == "title":
            return self.builder.build_title_plan_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"])
        raise ValueError("PLAN stage is not used for caption_mm")

    def _build_final_messages(self, df: pd.DataFrame, i: int, plan_json: str) -> List[Dict[str, str]]:
        if self.cfg.task == "two_words":
            return self.builder.build_two_words_final_messages(df.loc[i, "word1"], df.loc[i, "word2"], plan_json)
        if self.cfg.task == "title":
            return self.builder.build_title_final_messages(df.loc[i, "headline"], df.loc[i, "noun1"], df.loc[i, "noun2"], plan_json)
        if self.cfg.task == "caption_mm":
            return self.builder.build_caption_messages(df.loc[i, "scene"], df.loc[i, "noun1"], df.loc[i, "noun2"])
        raise ValueError(f"Unknown task: {self.cfg.task}")

    def _eval_dict(self, out_text: str, word1: str, word2: str) -> Dict[str, Any]:
        evaluator = self._ensure_evaluator()
        required_ok = evaluator.required_words_present(out_text, word1, word2)
        humorous_p = evaluator.humor_prob(out_text)
        humorous_ok = humorous_p > 0.5

        extra = {}
        if self.cfg.task == "caption_mm":
            short_ok = evaluator.is_short_caption(out_text, max_words=self.cfg.caption_max_words)
            extra["caption"] = {"short_enough": bool(short_ok), "max_words": int(self.cfg.caption_max_words)}

        good_ok = bool(required_ok and humorous_ok and (extra.get("caption", {}).get("short_enough", True)))
        return {
            "required_words": {"anchor1": word1, "anchor2": word2, "present": bool(required_ok)},
            "humor_classifier": {"p_humor": float(humorous_p), "humorous": bool(humorous_ok)},
            "overall": {"good": bool(good_ok)},
            **extra,
        }

    def _is_good(self, df: pd.DataFrame, i: int, out: str) -> bool:
        out = normalize_one_line(out)
        if out == "":
            return False
        word1, word2 = self._anchors_for_row(df, i)
        return bool(self._eval_dict(out, word1, word2)["overall"]["good"])

    # ---------------------------
    # caption_mm: K candidates + rerank
    # ---------------------------

    def _generate_best_caption_for_rows(
        self,
        *,
        model,
        tokenizer,
        df: pd.DataFrame,
        row_ids: List[int],
        decode: DecodeConfig,
        k: int,
    ) -> Dict[int, str]:
        """
        For each row in row_ids:
          - generate k candidates (by repeating the same prompt k times)
          - choose the best valid candidate by humor probability
          - if none valid, return the best-by-humor anyway (still useful for retry rounds)
        """
        evaluator = self._ensure_evaluator()
        results: Dict[int, str] = {}

        # Build repeated prompts
        prompts: List[str] = []
        idx_map: List[int] = []

        for rid in row_ids:
            msgs = self._build_final_messages(df, rid, plan_json="")
            chat = to_chat_text(tokenizer, msgs)
            for _ in range(max(1, k)):
                prompts.append(chat)
                idx_map.append(rid)

        outs = generate_batch_once(model, tokenizer, prompts, decode)

        # Group candidates per row
        buckets: Dict[int, List[str]] = {}
        for rid, text in zip(idx_map, outs):
            buckets.setdefault(rid, []).append(normalize_one_line(text))

        # Pick best
        for rid in row_ids:
            cands = buckets.get(rid, [])
            a1, a2 = self._anchors_for_row(df, rid)
            a1 = safe_word(a1) or a1
            a2 = safe_word(a2) or a2

            best_valid = ("", -1.0)
            best_any = ("", -1.0)

            for c in cands:
                if not c:
                    continue
                p = evaluator.humor_prob(c)
                if p > best_any[1]:
                    best_any = (c, p)

                valid, score = evaluator.score_caption_candidate(
                    c, a1, a2, max_words=self.cfg.caption_max_words
                )
                if valid and score > best_valid[1]:
                    best_valid = (c, score)

            chosen = best_valid[0] if best_valid[1] >= 0 else best_any[0]
            results[rid] = normalize_one_line(chosen)

        return results

    # ---------------------------
    # caption_mm: prepass (VL)
    # ---------------------------

    def _caption_mm_prepare_scene_and_anchors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns: scene, noun1, noun2
        """
        print("\nStage: VL PREPASS (GIF -> scene + two nouns)")
        vl = VLSceneExtractor()
        cache_dir = self.cfg.output_dir / "_gif_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        scenes: List[str] = []
        n1s: List[str] = []
        n2s: List[str] = []

        n = len(df)
        for i in range(n):
            url = str(df.loc[i, "gif_url"]).strip()
            # Pick GIF source: prefer local path, otherwise use URL (downloaded earlier or to temp)
            local = ""
            if "gif_path" in df.columns:
                local = str(df.loc[i, "gif_path"]).strip()

            if not local:
                # fall back to gif_url (alias of url)
                if "gif_url" in df.columns:
                    local = str(df.loc[i, "gif_url"]).strip()

            if not local:
                raise ValueError("caption_mm row has neither gif_path nor gif_url")
            # If it's a URL, download to a temp GIF path for frame extraction
            if local.startswith("http://") or local.startswith("https://"):
                from pathlib import Path as _Path
                import requests as _requests

                tmp_dir = _Path(self.cfg.output_dir) / "_tmp_gifs"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_gif = tmp_dir / f"{str(df.loc[i, 'id'])}.gif"

                if not tmp_gif.exists() or tmp_gif.stat().st_size == 0:
                    rr = _requests.get(local, timeout=60)
                    rr.raise_for_status()
                    tmp_gif.write_bytes(rr.content)

                local = str(tmp_gif)


            if local:
                gif_path = Path(local)
            else:
                gif_path = cache_dir / f"{i}.gif"
                if url:
                    download_gif(url, gif_path, timeout_seconds=self.cfg.gif_frames.timeout_seconds)
                else:
                    scenes.append("A short clip showing a simple situation.")
                    n1s.append("object")
                    n2s.append("person")
                    continue

            frames = extract_frames(
                gif_path,
                max_frames=self.cfg.gif_frames.max_frames,
                max_side=self.cfg.gif_frames.max_side,
            )

            scene, (a1, a2) = vl.extract_scene_and_anchors(frames)
            scenes.append(scene)
            n1s.append(a1)
            n2s.append(a2)

            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"  vl: {i + 1}/{n}")

        df = df.copy()
        df["scene"] = scenes
        df["noun1"] = n1s
        df["noun2"] = n2s
        return df

    # ---------------------------
    # main
    # ---------------------------

    def run(self) -> None:
        configure_fast_kernels()

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cfg.output_dir / self.cfg.output_filename
        log_path = out_path.with_suffix(out_path.suffix + ".log.json")

        print("Project root:", self.builder.config.paths.project_root)
        print("Using input :", self.cfg.input_path)
        print("Using output:", out_path)
        print("Using log   :", log_path)

        df = self._load_dataframe()
        n = len(df)
        print(f"Loaded {n} rows.")

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
                "candidates_per_row": self.cfg.candidates_per_row,
                "caption_max_words": self.cfg.caption_max_words,
                "drive_output_dir": self.cfg.drive_output_dir,
            }
        )

        # Caption MM: VL prepass creates noun1/noun2/scene
        if self.cfg.task == "caption_mm":
            df = self._caption_mm_prepare_scene_and_anchors(df)

        print("Loading base model + tokenizer...")
        model, tokenizer = load_model_and_tokenizer(self.cfg.base_model_id, getattr(self.cfg, 'lora_adapter_path', None))

        # Per-row metadata for logging
        metas: List[Dict[str, Any]] = []
        for i in range(n):
            meta: Dict[str, Any] = {"anchors": {"word1": "", "word2": ""}, "headline": None, "plan": None}
            if self.cfg.task == "two_words":
                meta["anchors"]["word1"] = str(df.loc[i, "word1"])
                meta["anchors"]["word2"] = str(df.loc[i, "word2"])
            elif self.cfg.task == "title":
                meta["headline"] = str(df.loc[i, "headline"])
                meta["anchors"]["word1"] = str(df.loc[i, "noun1"])
                meta["anchors"]["word2"] = str(df.loc[i, "noun2"])
            elif self.cfg.task == "caption_mm":
                meta["anchors"]["word1"] = str(df.loc[i, "noun1"])
                meta["anchors"]["word2"] = str(df.loc[i, "noun2"])
                meta["scene"] = str(df.loc[i, "scene"])
            metas.append(meta)

        indices = list(range(n))

        # ---------------------------------------------------------------------
        # PLAN stage (skip for caption_mm)
        # ---------------------------------------------------------------------
        plans: List[str] = [""] * n
        if self.cfg.task != "caption_mm":
            print("\nStage: PLAN")
            t_plan = time.time()

            for batch_index, batch_ids in enumerate(batched(indices, self.cfg.batch_size), start=1):
                batch_texts: List[str] = []
                for i in batch_ids:
                    msgs = self._build_plan_messages(df, i)
                    batch_texts.append(to_chat_text(tokenizer, msgs))

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

                done = min(batch_index * self.cfg.batch_size, n)
                dt = time.time() - t_plan
                print(f"  plan batches: {batch_index}  ({done}/{n}, elapsed {dt:.1f}s)")

            print(f"PLAN done in {time.time() - t_plan:.1f}s")

        # ---------------------------------------------------------------------
        # FINAL stage
        # ---------------------------------------------------------------------
        print("\nStage: FINAL")
        outputs: List[str] = [""] * n
        used_fallback: List[bool] = [False] * n
        last_before_fallback: List[str] = [""] * n
        ever_failed: set[int] = set()

        t_final = time.time()

        if self.cfg.task == "caption_mm":
            # caption_mm: K candidates + select
            for batch_index, batch_ids in enumerate(batched(indices, self.cfg.batch_size), start=1):
                chosen = self._generate_best_caption_for_rows(
                    model=model,
                    tokenizer=tokenizer,
                    df=df,
                    row_ids=batch_ids,
                    decode=self.cfg.final_decode,
                    k=max(1, self.cfg.candidates_per_row),
                )

                for i in batch_ids:
                    out = normalize_one_line(chosen.get(i, ""))
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

                    # log the first prompt in batch
                    msgs0 = self._build_final_messages(df, i0, plan_json="")
                    chat0 = to_chat_text(tokenizer, msgs0)
                    logger.log_first_in_batch(
                        stage="FINAL",
                        batch_index=batch_index,
                        batch_size=len(batch_ids),
                        example_index=i0,
                        chat_text=chat0,
                        model_output=outputs[i0],
                        meta=metas[i0],
                        evaluation=evaluation,
                        note=f"caption_mm k={self.cfg.candidates_per_row}",
                    )

                done = min(batch_index * self.cfg.batch_size, n)
                dt = time.time() - t_final
                print(f"  final batches: {batch_index}  ({done}/{n}, elapsed {dt:.1f}s)")

        else:
            # original behavior: one sample per row
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

                done = min(batch_index * self.cfg.batch_size, n)
                dt = time.time() - t_final
                print(f"  final batches: {batch_index}  ({done}/{n}, elapsed {dt:.1f}s)")

        bad_indices: List[int] = []
        for i in indices:
            if not self._is_good(df, i, outputs[i]):
                bad_indices.append(i)
                ever_failed.add(i)

        # ---------------------------------------------------------------------
        # Retries
        # ---------------------------------------------------------------------
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

            new_bad: List[int] = []
            t_retry = time.time()
            total = len(bad_indices)

            if self.cfg.task == "caption_mm":
                for batch_index, batch_ids in enumerate(batched(bad_indices, self.cfg.batch_size), start=1):
                    chosen = self._generate_best_caption_for_rows(
                        model=model,
                        tokenizer=tokenizer,
                        df=df,
                        row_ids=batch_ids,
                        decode=decode,
                        k=max(1, self.cfg.candidates_per_row),
                    )

                    for i in batch_ids:
                        out = normalize_one_line(chosen.get(i, ""))
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

                        msgs0 = self._build_final_messages(df, i0, plan_json="")
                        chat0 = to_chat_text(tokenizer, msgs0)

                        logger.log_first_in_batch(
                            stage=f"FINAL RETRY {retry_round}",
                            batch_index=batch_index,
                            batch_size=len(batch_ids),
                            example_index=i0,
                            chat_text=chat0,
                            model_output=outputs[i0],
                            meta=metas[i0],
                            evaluation=evaluation,
                            note=f"caption_mm k={self.cfg.candidates_per_row}",
                        )

                    for i in batch_ids:
                        if not self._is_good(df, i, outputs[i]):
                            new_bad.append(i)
                            ever_failed.add(i)

                    elapsed = time.time() - t_retry
                    done = min(batch_index * self.cfg.batch_size, total)
                    print(f"  retry {retry_round} batches: {batch_index}  ({done}/{total}, elapsed {elapsed:.1f}s)")

            else:
                # original retry path (single sample each)
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

                    elapsed = time.time() - t_retry
                    done = min(batch_index * self.cfg.batch_size, total)
                    print(f"  retry {retry_round} batches: {batch_index}  ({done}/{total}, elapsed {elapsed:.1f}s)")

            bad_indices = new_bad

        # ---------------------------------------------------------------------
        # Fallbacks
        # ---------------------------------------------------------------------
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
                        "scene": metas[i].get("scene"),
                        "final_prediction": outputs[i],
                    }
                )

        # ---------------------------------------------------------------------
        # Save predictions
        # ---------------------------------------------------------------------
        out_df = df.copy()
        if "id" not in out_df.columns:
            out_df.insert(0, "id", [str(i) for i in range(len(out_df))])

        out_df["prediction"] = outputs

        print(f"\nSaving predictions to: {out_path}")
        out_df.to_csv(out_path, sep="\t", index=False)

        # --- FINAL CAPTIONS PREVIEW (printed to stdout) ---
        try:
            show_n = 25
            cols = [c for c in ["id", "gif_url", "gif_path", "prediction"] if c in out_df.columns]
            print("\n=== FINAL CAPTIONS PREVIEW ===")
            for _, row in out_df[cols].head(show_n).iterrows():
                rid = str(row.get("id", "")).strip()
                pred = str(row.get("prediction", "")).strip()
                if rid:
                    print(f"{rid}\t{pred}")
                else:
                    print(pred)
            print("=== END PREVIEW ===\n")
        except Exception as _e:
            print("[preview] could not print captions:", _e)


        zip_path = zip_file(out_path)
        print(f"Zipping to: {zip_path}")

        # ---------------------------------------------------------------------
        # Log summaries
        # ---------------------------------------------------------------------
        failed_items: List[Dict[str, Any]] = []
        for i in sorted(ever_failed):
            w1, w2 = self._anchors_for_row(df, i)
            eval_final = None
            try:
                evaluator = self._ensure_evaluator()
                eval_final = {
                    "required_words_present": bool(evaluator.required_words_present(outputs[i], w1, w2)),
                    "p_humor": float(evaluator.humor_prob(outputs[i])),
                    "short_ok": bool(evaluator.is_short_caption(outputs[i], max_words=self.cfg.caption_max_words))
                    if self.cfg.task == "caption_mm" else None,
                    "good": bool(self._is_good(df, i, outputs[i])),
                }
            except Exception:
                eval_final = None

            failed_items.append(
                {
                    "index": int(i),
                    "anchors": {"word1": w1, "word2": w2},
                    "headline": metas[i].get("headline"),
                    "scene": metas[i].get("scene"),
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

        final_elapsed = float(time.time() - t_final)
        logger.log_run_end(
            {
                "final_seconds": round(final_elapsed, 3),
                "rows": n,
                "ever_failed_count": len(ever_failed),
                "fallback_count": int(sum(1 for x in used_fallback if x)),
                "zip_path": str(zip_path),
                "log_path": str(log_path),
            }
        )
        logger.close()

        copy_outputs_to_drive([zip_path, log_path], self.cfg.drive_output_dir)
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
