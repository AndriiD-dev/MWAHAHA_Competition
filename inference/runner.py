from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from peft import PeftModel
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
class HumorPolicy:
    # Defaults per mode
    default_two_words: str = "pun"
    default_headline: str = "satire"
    default_image: str = "irony"

    # When (retry_round >= threshold) switch humor type
    two_words_switch_after: int = 3         # pun -> irony
    headline_switch_after: int = 2          # satire -> irony
    image_pun_fallback_after: int = 1       # pun -> irony

    # Conditional pun for image: markers hinting that text/labels appear
    image_text_markers: Tuple[str, ...] = (
        "sign", "label", "caption", "text", "words", "logo", "shirt", "screen", "poster", "menu"
    )
    image_written_markers: Tuple[str, ...] = ("says", "written", "printed", "reads")


@dataclass(frozen=True)
class RunnerConfig:
    # Supported tasks:
    # - "two_words"
    # - "headline"   (alias: "title")
    # - "image_caption_b1"
    # - "image_caption_b2"
    task: str

    base_model_id: str
    lora_adapter_path: Optional[str]

    input_path: Path
    output_dir: Path
    output_filename: str

    batch_size: int = 16

    plan_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=80, min_new_tokens=16, temperature=0.4, top_p=0.9
    )
    final_decode: DecodeConfig = DecodeConfig(
        max_new_tokens=64, min_new_tokens=10, temperature=0.9, top_p=0.95
    )

    max_retries: int = 5
    retry_steps: Tuple[RetryStep, ...] = (
        RetryStep(0.90, 0.95, 40),
        RetryStep(0.95, 0.98, 40),
        RetryStep(1.00, 0.98, 48),
        RetryStep(1.05, 0.98, 48),
        RetryStep(1.10, 0.99, 56),
    )

    # Headline noun selection
    noun_seed_base: int = 42

    # Image caption constraints
    caption_max_words: int = 20

    # Where the vision language extractor should write scenes
    scenes_output_csv: Optional[Path] = None

    # Vision language extractor invocation
    vl_scene_extractor_module: str = "inference.utils.vl_scene_extractor"
    vl_force_rerun: bool = False

    # Humor selection rules
    humor_policy: HumorPolicy = HumorPolicy()

    # Headline classification markers
    headline_public_markers: Tuple[str, ...] = (
        "government", "minister", "president", "parliament", "senate", "congress",
        "election", "campaign", "policy", "law", "court", "supreme", "police",
        "war", "military",
        "company", "corporate", "ceo", "shareholders", "platform", "social media",
        "artificial intelligence", "technology", "crypto", "bitcoin",
        "stocks", "market", "inflation", "economy",
        "study", "research", "scientists", "report",
    )
    headline_personal_markers: Tuple[str, ...] = (
        "student", "teacher", "family", "parents", "mom", "dad",
        "man", "woman", "kid", "child", "baby", "neighbor", "customer",
        "restaurant", "flight", "hotel", "home", "school", "office",
    )

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


def _humor_policy_from_dict(d: Dict[str, Any]) -> HumorPolicy:
    return HumorPolicy(
        default_two_words=str(d.get("default_two_words", "pun")),
        default_headline=str(d.get("default_headline", "satire")),
        default_image=str(d.get("default_image", "irony")),
        two_words_switch_after=int(d.get("two_words_switch_after", 3)),
        headline_switch_after=int(d.get("headline_switch_after", 2)),
        image_pun_fallback_after=int(d.get("image_pun_fallback_after", 1)),
        image_text_markers=tuple(d.get("image_text_markers", HumorPolicy.image_text_markers)),
        image_written_markers=tuple(d.get("image_written_markers", HumorPolicy.image_written_markers)),
    )


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
    lora_adapter_path = d.get("lora_adapter_path")
    if lora_adapter_path is not None:
        lora_adapter_path = str(lora_adapter_path)

    input_path = _as_path(d["input_path"])
    output_dir = _as_path(d["output_dir"])
    output_filename = str(d["output_filename"])

    batch_size = int(d.get("batch_size", 16))
    noun_seed_base = int(d.get("noun_seed_base", 42))
    caption_max_words = int(d.get("caption_max_words", 20))

    scenes_output_csv = d.get("scenes_output_csv")
    scenes_output_csv = _as_path(scenes_output_csv) if scenes_output_csv else None

    vl_scene_extractor_module = str(d.get("vl_scene_extractor_module", "inference.utils.vl_scene_extractor"))
    vl_force_rerun = bool(d.get("vl_force_rerun", False))

    plan_decode = _decode_from_dict(d.get("plan_decode", {}))
    final_decode = _decode_from_dict(d.get("final_decode", {}))

    retry_steps_cfg = d.get("retry_steps")
    retry_steps = _retry_steps_from_list(retry_steps_cfg) if retry_steps_cfg is not None else RunnerConfig.retry_steps
    max_retries = int(d.get("max_retries", 5))

    humor_policy_cfg = d.get("humor_policy") or {}
    humor_policy = _humor_policy_from_dict(humor_policy_cfg)

    headline_public_markers = tuple(d.get("headline_public_markers", RunnerConfig.headline_public_markers))
    headline_personal_markers = tuple(d.get("headline_personal_markers", RunnerConfig.headline_personal_markers))

    drive_output_dir = d.get("drive_output_dir")

    return RunnerConfig(
        task=task,
        base_model_id=base_model_id,
        lora_adapter_path=lora_adapter_path,
        input_path=input_path,
        output_dir=output_dir,
        output_filename=output_filename,
        batch_size=batch_size,
        plan_decode=plan_decode,
        final_decode=final_decode,
        max_retries=max_retries,
        retry_steps=retry_steps,
        noun_seed_base=noun_seed_base,
        caption_max_words=caption_max_words,
        scenes_output_csv=scenes_output_csv,
        vl_scene_extractor_module=vl_scene_extractor_module,
        vl_force_rerun=vl_force_rerun,
        humor_policy=humor_policy,
        headline_public_markers=headline_public_markers,
        headline_personal_markers=headline_personal_markers,
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


def load_model_and_tokenizer(model_id: str, lora_adapter_path: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else tokenizer.unk_token

    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=dtype if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
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
# Runner
# =============================================================================


Mode = str  # "two_words" | "headline" | "image_caption_b1" | "image_caption_b2"
HumorType = str  # "pun" | "irony" | "satire"


@dataclass
class InferenceRunner:
    cfg: RunnerConfig
    builder: PromptBuilder = field(default_factory=PromptBuilder)
    response_evaluator: Optional[ResponseEvaluator] = field(default=None, init=False)

    def _ensure_evaluator(self) -> ResponseEvaluator:
        if self.response_evaluator is None:
            self.response_evaluator = ResponseEvaluator()
        return self.response_evaluator

    def _canonical_mode(self) -> Mode:
        t = (self.cfg.task or "").strip().lower()
        if t == "title":
            return "headline"
        if t in {"two_words", "headline", "image_caption_b1", "image_caption_b2"}:
            return t
        raise ValueError(f"Unknown task: {self.cfg.task}")

    # ------------------------------------------------------------------
    # Humor strategy (config driven)
    # ------------------------------------------------------------------

    def _choose_humor_for_two_words(self) -> HumorType:
        return self.cfg.humor_policy.default_two_words

    def _choose_humor_for_headline(self, headline: str) -> HumorType:
        h = normalize_one_line(headline).lower()
        if not h:
            return self.cfg.humor_policy.default_headline

        if any(m in h for m in self.cfg.headline_public_markers):
            return self.cfg.humor_policy.default_headline

        if any(m in h for m in self.cfg.headline_personal_markers):
            return "irony"

        return self.cfg.humor_policy.default_headline

    def _should_try_pun_for_image(self, image_facts: str, noun1: str, noun2: str) -> bool:
        t = normalize_one_line(image_facts).lower()
        nouns = {safe_word(noun1).lower(), safe_word(noun2).lower()}

        if any(x in t for x in self.cfg.humor_policy.image_written_markers):
            return True
        if any(m in t for m in self.cfg.humor_policy.image_text_markers):
            return True
        if any(n in nouns for n in self.cfg.humor_policy.image_text_markers):
            return True
        return False

    def _choose_humor_for_image(self, image_facts: str, noun1: str, noun2: str) -> HumorType:
        if self._should_try_pun_for_image(image_facts, noun1, noun2):
            return "pun"
        return self.cfg.humor_policy.default_image

    def _maybe_switch_humor_on_retry(
        self,
        *,
        mode: Mode,
        current: HumorType,
        retry_round: int,
        image_facts: str = "",
        noun1: str = "",
        noun2: str = "",
    ) -> HumorType:
        hp = self.cfg.humor_policy

        if mode == "two_words":
            if current == "pun" and retry_round >= hp.two_words_switch_after:
                return "irony"
            return current

        if mode == "headline":
            if current == "satire" and retry_round >= hp.headline_switch_after:
                return "irony"
            return current

        if mode in {"image_caption_b1", "image_caption_b2"}:
            if current == "pun" and retry_round >= hp.image_pun_fallback_after:
                return "irony"
            return current

        return current

    # ------------------------------------------------------------------
    # Vision language scenes: shell command + wait
    # ------------------------------------------------------------------

    def _run_vl_scene_extractor(self, *, input_tsv: Path, output_csv: Path) -> None:
        cmd = [
            sys.executable,
            "-m",
            self.cfg.vl_scene_extractor_module,
            "--input_tsv",
            str(input_tsv),
            "--output_csv",
            str(output_csv),
        ]
        print("Running scene extraction:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _ensure_scenes_csv(self, *, mode: Mode) -> Path:
        if mode not in {"image_caption_b1", "image_caption_b2"}:
            raise ValueError("Scenes file is only used for image caption modes.")

        if self.cfg.scenes_output_csv is None:
            raise ValueError("Config must set scenes_output_csv for image caption modes.")

        out_csv = self.cfg.scenes_output_csv

        if self.cfg.vl_force_rerun and out_csv.exists():
            try:
                out_csv.unlink()
            except Exception:
                pass

        if out_csv.exists() and out_csv.stat().st_size > 0:
            try:
                tmp = pd.read_csv(out_csv, keep_default_na=False)
                if {"id", "scene", "noun1", "noun2"}.issubset(set(tmp.columns)):
                    return out_csv
            except Exception:
                pass

        self._run_vl_scene_extractor(input_tsv=self.cfg.input_path, output_csv=out_csv)
        return out_csv

    # ------------------------------------------------------------------
    # Loading data per mode
    # ------------------------------------------------------------------

    def _load_dataframe(self) -> Tuple[pd.DataFrame, Mode]:
        mode = self._canonical_mode()

        if mode == "two_words":
            df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
            need = {"word1", "word2"}
            missing = need.difference(set(df.columns))
            if missing:
                raise ValueError(f"Missing required columns in {self.cfg.input_path}: {sorted(missing)}")
            df["word1"] = df["word1"].fillna("").astype(str)
            df["word2"] = df["word2"].fillna("").astype(str)
            return df, mode

        if mode == "headline":
            try:
                df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
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

            print("Preparing planning nouns for each headline (spaCy noun extraction)...")
            noun1_list: List[str] = []
            noun2_list: List[str] = []
            for i in range(len(df)):
                seed = int(self.cfg.noun_seed_base + i)
                n1, n2 = self.builder.choose_two_nouns_from_headline(df.loc[i, "headline"], seed=seed)
                noun1_list.append(n1)
                noun2_list.append(n2)
            df["noun1"] = noun1_list
            df["noun2"] = noun2_list
            return df, mode

        if mode in {"image_caption_b1", "image_caption_b2"}:
            df = pd.read_csv(self.cfg.input_path, sep="\t", keep_default_na=False)
            if "id" not in df.columns:
                raise ValueError(f"Image input must have an 'id' column: {self.cfg.input_path}")

            scenes_csv = self._ensure_scenes_csv(mode=mode)
            scenes = pd.read_csv(scenes_csv, keep_default_na=False)

            need = {"id", "scene", "noun1", "noun2"}
            missing = need.difference(set(scenes.columns))
            if missing:
                raise ValueError(f"Scenes file missing columns {sorted(missing)}: {scenes_csv}")

            merged = df.merge(scenes[["id", "scene", "noun1", "noun2"]], on="id", how="left")
            merged["scene"] = merged["scene"].fillna("").astype(str)
            merged["noun1"] = merged["noun1"].fillna("").astype(str)
            merged["noun2"] = merged["noun2"].fillna("").astype(str)
            merged["image_facts"] = merged["scene"]

            if mode == "image_caption_b2":
                if "prompt_text" not in merged.columns:
                    if "prompt" in merged.columns:
                        merged["prompt_text"] = merged["prompt"]
                if "prompt_text" not in merged.columns:
                    raise ValueError("Variant two requires column 'prompt' or 'prompt_text'.")
                merged["prompt_text"] = merged["prompt_text"].fillna("").astype(str)

            return merged, mode

        raise ValueError(f"Unsupported mode: {mode}")

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def _build_plan_messages(self, df: pd.DataFrame, i: int, *, mode: Mode, humor_type: HumorType) -> List[Dict[str, str]]:
        if mode == "two_words":
            return self.builder.build_plan_messages(
                mode="two_words",
                humor_type=humor_type,
                noun1=str(df.loc[i, "word1"]),
                noun2=str(df.loc[i, "word2"]),
            )
        if mode == "headline":
            return self.builder.build_plan_messages(
                mode="headline",
                humor_type=humor_type,
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                headline=str(df.loc[i, "headline"]),
            )
        if mode == "image_caption_b1":
            return self.builder.build_plan_messages(
                mode="image_caption_b1",
                humor_type=humor_type,
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                image_facts=str(df.loc[i, "image_facts"]),
            )
        if mode == "image_caption_b2":
            return self.builder.build_plan_messages(
                mode="image_caption_b2",
                humor_type=humor_type,
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                image_facts=str(df.loc[i, "image_facts"]),
                prompt_text=str(df.loc[i, "prompt_text"]),
            )
        raise ValueError(f"Unsupported mode: {mode}")

    def _build_final_messages(self, df: pd.DataFrame, i: int, *, mode: Mode, plan_json: str) -> List[Dict[str, str]]:
        if mode == "two_words":
            return self.builder.build_final_messages(
                mode="two_words",
                noun1=str(df.loc[i, "word1"]),
                noun2=str(df.loc[i, "word2"]),
                plan_text=plan_json,
            )
        if mode == "headline":
            return self.builder.build_final_messages(
                mode="headline",
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                headline=str(df.loc[i, "headline"]),
                plan_text=plan_json,
            )
        if mode == "image_caption_b1":
            return self.builder.build_final_messages(
                mode="image_caption_b1",
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                image_facts=str(df.loc[i, "image_facts"]),
                plan_text=plan_json,
            )
        if mode == "image_caption_b2":
            return self.builder.build_final_messages(
                mode="image_caption_b2",
                noun1=str(df.loc[i, "noun1"]),
                noun2=str(df.loc[i, "noun2"]),
                image_facts=str(df.loc[i, "image_facts"]),
                prompt_text=str(df.loc[i, "prompt_text"]),
                plan_text=plan_json,
            )
        raise ValueError(f"Unsupported mode: {mode}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _count_words(self, text: str) -> int:
        t = normalize_one_line(text)
        if not t:
            return 0
        return len([w for w in t.split(" ") if w])

    def _copies_headline_verbatim(self, out: str, headline: str) -> bool:
        o = normalize_one_line(out).lower()
        h = normalize_one_line(headline).lower()
        return bool(h) and (h in o)

    def _is_good(self, df: pd.DataFrame, i: int, *, mode: Mode, out: str) -> bool:
        out = normalize_one_line(out)
        if out == "":
            return False

        evaluator = self._ensure_evaluator()
        try:
            humorous_ok = bool(evaluator.is_humorous(out))
        except Exception:
            humorous_ok = True

        if mode == "two_words":
            w1 = str(df.loc[i, "word1"])
            w2 = str(df.loc[i, "word2"])
            required_ok = bool(evaluator.required_words_present(out, w1, w2))
            return bool(required_ok and humorous_ok)

        if mode == "headline":
            headline = str(df.loc[i, "headline"])
            if self._copies_headline_verbatim(out, headline):
                return False
            return bool(humorous_ok)

        if mode == "image_caption_b1":
            return bool(humorous_ok and (self._count_words(out) <= self.cfg.caption_max_words))

        if mode == "image_caption_b2":
            pt = normalize_one_line(str(df.loc[i, "prompt_text"]))
            if pt and not out.startswith(pt):
                return False
            return bool(humorous_ok and (self._count_words(out) <= self.cfg.caption_max_words))

        return False

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_plan(self, df: pd.DataFrame, i: int, *, mode: Mode, humor_type: HumorType) -> str:
        if mode == "two_words":
            w1 = safe_word(str(df.loc[i, "word1"]))
            w2 = safe_word(str(df.loc[i, "word2"]))
            return json.dumps({"humor_type": humor_type, "mode": mode, "nouns": [w1, w2]}, ensure_ascii=False)

        if mode == "headline":
            h = normalize_one_line(str(df.loc[i, "headline"]))[:160]
            n1 = safe_word(str(df.loc[i, "noun1"]))
            n2 = safe_word(str(df.loc[i, "noun2"]))
            return json.dumps({"humor_type": humor_type, "mode": mode, "headline": h, "nouns": [n1, n2]}, ensure_ascii=False)

        if mode in {"image_caption_b1", "image_caption_b2"}:
            img = normalize_one_line(str(df.loc[i, "image_facts"]))[:160]
            n1 = safe_word(str(df.loc[i, "noun1"]))
            n2 = safe_word(str(df.loc[i, "noun2"]))
            payload: Dict[str, Any] = {"humor_type": humor_type, "mode": mode, "nouns": [n1, n2], "image_facts": img}
            if mode == "image_caption_b2":
                payload["prompt_text"] = normalize_one_line(str(df.loc[i, "prompt_text"]))[:80]
            return json.dumps(payload, ensure_ascii=False)

        return json.dumps({"humor_type": humor_type, "mode": mode}, ensure_ascii=False)

    def _fallback_output(self, df: pd.DataFrame, i: int, *, mode: Mode) -> str:
        if mode == "two_words":
            w1 = safe_word(str(df.loc[i, "word1"])) or "word"
            w2 = safe_word(str(df.loc[i, "word2"])) or "word"
            return normalize_one_line(f"I tried to use {w1}, but {w2} immediately proved me wrong.")

        if mode == "headline":
            n1 = safe_word(str(df.loc[i, "noun1"])) or "news"
            return normalize_one_line(f"Breaking: Experts say this {n1} situation is totally normal and definitely under control.")

        if mode == "image_caption_b1":
            return normalize_one_line("Nothing to see here. Absolutely flawless execution.")

        if mode == "image_caption_b2":
            pt = normalize_one_line(str(df.loc[i, "prompt_text"]))
            out = (pt + " and I still call it a win.") if pt else "This is fine, and I still call it a win."
            return " ".join(out.split()[: self.cfg.caption_max_words])

        return "Just a joke."

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        configure_fast_kernels()

        mode = self._canonical_mode()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.cfg.output_dir / self.cfg.output_filename
        log_path = out_path.with_suffix(out_path.suffix + ".log.json")

        df, mode = self._load_dataframe()
        n = len(df)

        logger = Logger(enabled=True, log_path=log_path, max_chars=900)
        logger.log_run_meta(
            {
                "task": self.cfg.task,
                "mode": mode,
                "base_model_id": self.cfg.base_model_id,
                "lora_adapter_path": self.cfg.lora_adapter_path,
                "input_path": str(self.cfg.input_path),
                "output_dir": str(self.cfg.output_dir),
                "output_filename": self.cfg.output_filename,
                "batch_size": self.cfg.batch_size,
                "caption_max_words": self.cfg.caption_max_words,
                "scenes_output_csv": str(self.cfg.scenes_output_csv) if self.cfg.scenes_output_csv else None,
                "vl_scene_extractor_module": self.cfg.vl_scene_extractor_module,
                "vl_force_rerun": self.cfg.vl_force_rerun,
            }
        )

        model, tokenizer = load_model_and_tokenizer(self.cfg.base_model_id, self.cfg.lora_adapter_path)

        # Decide humor type per row
        metas: List[Dict[str, Any]] = []
        humor_types: List[HumorType] = [""] * n

        for i in range(n):
            meta: Dict[str, Any] = {"mode": mode, "humor_type": "", "headline": None, "plan": None}

            if mode == "two_words":
                ht = self._choose_humor_for_two_words()
            elif mode == "headline":
                ht = self._choose_humor_for_headline(str(df.loc[i, "headline"]))
                meta["headline"] = str(df.loc[i, "headline"])
            elif mode in {"image_caption_b1", "image_caption_b2"}:
                ht = self._choose_humor_for_image(str(df.loc[i, "image_facts"]), str(df.loc[i, "noun1"]), str(df.loc[i, "noun2"]))
            else:
                ht = "irony"

            humor_types[i] = ht
            meta["humor_type"] = ht
            metas.append(meta)

        indices = list(range(n))

        # ---------------------------
        # PLAN stage
        # ---------------------------
        plans: List[str] = [""] * n
        for batch_ids in batched(indices, self.cfg.batch_size):
            batch_texts: List[str] = []
            for i in batch_ids:
                msgs = self._build_plan_messages(df, i, mode=mode, humor_type=humor_types[i])
                batch_texts.append(to_chat_text(tokenizer, msgs))

            batch_out = generate_batch_once(model, tokenizer, batch_texts, self.cfg.plan_decode)

            for local, i in enumerate(batch_ids):
                p = normalize_one_line(batch_out[local])
                if not p:
                    p = self._fallback_plan(df, i, mode=mode, humor_type=humor_types[i])
                plans[i] = p
                metas[i]["plan"] = logger.try_parse_json(p) or p

        # ---------------------------
        # FINAL stage + retries
        # ---------------------------
        outputs: List[str] = [""] * n
        used_fallback: List[bool] = [False] * n
        last_before_fallback: List[str] = [""] * n
        ever_failed: set[int] = set()

        def generate_final(batch_ids: List[int], decode: DecodeConfig) -> None:
            batch_texts: List[str] = []
            for i in batch_ids:
                msgs = self._build_final_messages(df, i, mode=mode, plan_json=plans[i])
                batch_texts.append(to_chat_text(tokenizer, msgs))
            batch_out = generate_batch_once(model, tokenizer, batch_texts, decode)
            for local, i in enumerate(batch_ids):
                out = normalize_one_line(batch_out[local])
                outputs[i] = out
                last_before_fallback[i] = out

        for batch_ids in batched(indices, self.cfg.batch_size):
            generate_final(batch_ids, self.cfg.final_decode)

        bad_indices = [i for i in indices if not self._is_good(df, i, mode=mode, out=outputs[i])]
        ever_failed.update(bad_indices)

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

            # Switch humor type on retry (config driven), then replan, then regenerate
            for i in bad_indices:
                humor_types[i] = self._maybe_switch_humor_on_retry(
                    mode=mode,
                    current=humor_types[i],
                    retry_round=retry_round,
                    image_facts=str(df.loc[i, "image_facts"]) if mode.startswith("image_") else "",
                    noun1=str(df.loc[i, "noun1"]) if mode.startswith("image_") else "",
                    noun2=str(df.loc[i, "noun2"]) if mode.startswith("image_") else "",
                )
                metas[i]["humor_type"] = humor_types[i]

            # replan
            for batch_ids in batched(bad_indices, self.cfg.batch_size):
                batch_texts: List[str] = []
                for i in batch_ids:
                    msgs = self._build_plan_messages(df, i, mode=mode, humor_type=humor_types[i])
                    batch_texts.append(to_chat_text(tokenizer, msgs))
                batch_out = generate_batch_once(model, tokenizer, batch_texts, self.cfg.plan_decode)
                for local, i in enumerate(batch_ids):
                    p = normalize_one_line(batch_out[local])
                    if p:
                        plans[i] = p
                        metas[i]["plan"] = logger.try_parse_json(p) or p

            # final
            for batch_ids in batched(bad_indices, self.cfg.batch_size):
                generate_final(batch_ids, decode)

            bad_indices = [i for i in bad_indices if not self._is_good(df, i, mode=mode, out=outputs[i])]
            ever_failed.update(bad_indices)

        if bad_indices:
            for i in bad_indices:
                used_fallback[i] = True
                outputs[i] = self._fallback_output(df, i, mode=mode)

        # ---------------------------
        # Save predictions
        # ---------------------------
        out_df = df.copy()
        if "id" not in out_df.columns:
            out_df.insert(0, "id", [str(i) for i in range(len(out_df))])

        # REQUIRED: humor_type column
        out_df["humor_type"] = humor_types
        out_df["prediction"] = outputs

        out_df.to_csv(out_path, sep="\t", index=False)

        zip_path = zip_file(out_path)
        logger.log_run_end(
            {
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
# Entry point
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

    InferenceRunner(cfg=cfg).run()


if __name__ == "__main__":
    main()
