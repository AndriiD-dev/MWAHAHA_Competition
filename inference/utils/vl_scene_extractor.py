from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image, ImageSequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import pandas as pd
import requests

from inference.config import PromptBuilderConfig
from inference.utils.prompt_builder import PromptBuilder


_WS = re.compile(r"\s+")
_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _one_line(s: str) -> str:
    return _WS.sub(" ", (s or "").replace("\n", " ").replace("\r", " ")).strip()


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        m = _JSON_OBJ_RE.search(s)
        if m:
            s = m.group(0).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _base_form(w: str) -> str:
    w = (w or "").strip().lower()
    # strip possessive-like (rare for VL but cheap)
    w = re.sub(r"(?:'s)$", "", w)
    # plural heuristics (similar to spacy_extractor baseform variants)
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
        return w[:-1]
    return w


def _facts_ok(text: Any, *, min_words: int = 8, max_words: int = 25) -> bool:
    if not isinstance(text, str):
        return False
    t = _one_line(text)

    # one sentence-ish: avoid multiple hard sentence splits
    if t.count(".") + t.count("!") + t.count("?") > 2:
        return False

    # avoid joke-y markers
    if re.search(r"\b(joke|funny|hilarious|lol|lmao)\b", t, flags=re.IGNORECASE):
        return False

    words = [w for w in re.split(r"\s+", t) if w]
    return min_words <= len(words) <= max_words



def _anchors_ok(a: Any, *, banned: Iterable[str]) -> bool:
    if not isinstance(a, list) or len(a) != 2:
        return False

    bset = {str(x).strip().lower() for x in banned if str(x).strip()}

    w1 = a[0] if len(a) > 0 else ""
    w2 = a[1] if len(a) > 1 else ""
    if not isinstance(w1, str) or not isinstance(w2, str):
        return False

    w1 = w1.strip().lower()
    w2 = w2.strip().lower()

    # must be single-token lowercase alphabetic (spacy_extractor enforces word-ish tokens) :contentReference[oaicite:6]{index=6}
    if not re.fullmatch(r"[a-z]+", w1) or not re.fullmatch(r"[a-z]+", w2):
        return False
    if len(w1) < 3 or len(w2) < 3:
        return False

    # ban generic/low-signal nouns
    if w1 in bset or w2 in bset:
        return False

    # avoid duplicates and near-duplicates (plural/base-form collapse)
    if w1 == w2:
        return False
    if _base_form(w1) == _base_form(w2):
        return False

    # reject contraction-like endings just in case (mirrors spacy_extractor reject_contraction_like) :contentReference[oaicite:7]{index=7}
    if re.search(r"'(?:ll|re|ve|m|d|s)$", w1) or re.search(r"'(?:ll|re|ve|m|d|s)$", w2):
        return False

    return True


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for want in candidates:
        k = want.lower()
        if k in cols:
            return cols[k]
    return ""


def _is_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _download(url: str, dest: Path, *, timeout_seconds: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, timeout=timeout_seconds)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _extract_gif_frames(
    gif_path: Path,
    *,
    max_frames: int = 6,
    max_side: int = 512,
) -> List[Image.Image]:
    """
    Simple, dependency-free frame extraction:
    - evenly samples up to max_frames frames from the GIF
    - converts frames to RGB
    - resizes so max(width,height) <= max_side
    """
    img = Image.open(gif_path)
    frames_all = [f.copy() for f in ImageSequence.Iterator(img)]
    if not frames_all:
        return []

    n = len(frames_all)
    if n <= max_frames:
        idxs = list(range(n))
    else:
        idxs = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]

    out: List[Image.Image] = []
    for i in idxs:
        fr = frames_all[int(i)].convert("RGB")
        w, h = fr.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            fr = fr.resize((int(w * scale), int(h * scale)))
        out.append(fr)
    return out


@dataclass
class VLSceneExtractor:
    """
    Vision-language preprocessing:
      - input: GIF frames
      - output: (scene, (noun1, noun2))

    Also supports:
      - input tab separated values -> output comma separated values with scene + nouns
    """
    config: PromptBuilderConfig = field(default_factory=PromptBuilderConfig)

    _processor: Optional[Any] = field(default=None, init=False, repr=False)
    _model: Optional[Any] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "VLSceneExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------
    # Model lifecycle
    # ----------------------------

    def _load(self) -> Tuple[Any, Any]:
        s = self.config.vl_scene_extractor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(s.model_id, trust_remote_code=True)

        if self._model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                s.model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=dtype if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation=s.attn_implementation,
            )
            self._model.eval()
            try:
                self._model.generation_config.pad_token_id = self._processor.tokenizer.pad_token_id
            except Exception:
                pass

        return self._model, self._processor

    def close(self) -> None:
        """
        Release model and processor references and clear caches.
        This frees most graphics processing unit memory and helps free random access memory.
        """
        try:
            if self._model is not None:
                try:
                    # Best effort: move to central processing unit before deletion
                    self._model.to("cpu")
                except Exception:
                    pass
                del self._model
            if self._processor is not None:
                del self._processor
        finally:
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------------------
    # Generation + parsing
    # ----------------------------

    def _gen(self, frames: List[Image.Image], system: str, user: str) -> str:
        s = self.config.vl_scene_extractor
        model, processor = self._load()

        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": user}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=int(s.max_new_tokens),
                do_sample=True,
                temperature=float(s.temperature),
                top_p=float(s.top_p),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        prompt_len = int(inputs["input_ids"].shape[1])
        gen_only = out_ids[:, prompt_len:]
        txt = processor.batch_decode(gen_only, skip_special_tokens=True)[0]
        return _one_line(txt)

    def extract_scene_and_anchors(
        self,
        frames: List[Image.Image],
        *,
        max_tries: Optional[int] = None,
    ) -> Tuple[str, Tuple[str, str]]:
        """
        Returns:
          scene: a short neutral description (aligned to plan prompt name "image_facts")
          noun1, noun2: two concrete visible nouns, lowercase
        """
        s = self.config.vl_scene_extractor
        tries_limit = int(max_tries) if max_tries is not None else int(s.max_tries)

        banned = {x.lower() for x in self.config.generic_nouns}
        # Extra hard bans for image extraction quality
        banned |= {x.lower() for x in self.config.vl_scene_extractor.banned_nouns_extra}

        system = s.system_prompt
        user = s.user_prompt

        raw = self._gen(frames, system, user)
        obj = _try_parse_json(raw)

        tries = 1
        while tries < tries_limit:
            if obj:
                facts_val = obj.get(s.json_facts_key) or obj.get("scene")
                nouns_val = obj.get(s.json_nouns_key) or obj.get("anchors")
                if _facts_ok(facts_val) and _anchors_ok(nouns_val, banned=banned):
                    break

            fix_user = s.repair_user_template.format(previous=raw)
            raw = self._gen(frames, system, fix_user)
            obj = _try_parse_json(raw)
            tries += 1

        if not obj:
            return ("A short clip showing a simple situation.", ("object", "person"))

        facts_val = obj.get(s.json_facts_key) or obj.get("scene")
        nouns_val = obj.get(s.json_nouns_key) or obj.get("anchors")

        if not _facts_ok(facts_val) or not _anchors_ok(nouns_val, banned=banned):
            return ("A short clip showing a simple situation.", ("object", "person"))

        scene = _one_line(str(facts_val))
        a1 = str(nouns_val[0]).strip().lower()
        a2 = str(nouns_val[1]).strip().lower()
        return (scene, (a1, a2))

    # ----------------------------
    # TSV -> CSV preprocessing
    # ----------------------------

    def extract_tsv_to_csv(
        self,
        *,
        input_tsv: Path,
        output_csv: Optional[Path] = None,
        max_frames: int = 6,
        max_side: int = 512,
        unload_after: bool = True,
        download_timeout_seconds: int = 60,
    ) -> Path:
        """
        Reads a tab separated values file with a GIF location column.
        Writes a comma separated values file with extracted scene + nouns.

        Output includes:
          - id (existing or created)
          - gif_url and or gif_path (if present)
          - prompt_text (if present)
          - scene, noun1, noun2 (new)
        """
        df = pd.read_csv(input_tsv, sep="\t", keep_default_na=False)

        s = self.config.vl_scene_extractor

        id_col = _pick_col(df, [getattr(s, "id_col", "id"), "id"])
        gif_url_col = _pick_col(df, [getattr(s, "gif_url_col", "gif_url"), "gif_url", "url"])
        gif_path_col = _pick_col(df, [getattr(s, "gif_path_col", "gif_path"), "gif_path", "path", "filepath", "file"])
        prompt_text_col = _pick_col(df, [getattr(s, "prompt_text_col", "prompt_text"), "prompt_text", "prompt", "text"])

        if not id_col:
            df.insert(0, "id", [f"row_{i:05d}" for i in range(len(df))])
            id_col = "id"

        if not gif_url_col and not gif_path_col:
            raise ValueError("Input must contain a GIF url column (gif_url or url) or a GIF path column (gif_path).")

        out = output_csv
        if out is None:
            out = self.config.paths.data_dir / f"{Path(input_tsv).stem}.scenes.csv"
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp_dir = self.config.paths.data_dir / "_tmp_gifs"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        scenes: List[str] = []
        n1s: List[str] = []
        n2s: List[str] = []

        n = len(df)
        try:
            for i in range(n):
                rid = str(df.loc[i, id_col]).strip() or f"row_{i:05d}"

                local = ""
                if gif_path_col:
                    local = str(df.loc[i, gif_path_col]).strip()

                if not local and gif_url_col:
                    local = str(df.loc[i, gif_url_col]).strip()

                if not local:
                    scenes.append("A short clip showing a simple situation.")
                    n1s.append("object")
                    n2s.append("person")
                    continue

                if _is_url(local):
                    gif_path = _download(local, tmp_dir / f"{rid}.gif", timeout_seconds=download_timeout_seconds)
                else:
                    gif_path = Path(local)

                frames = _extract_gif_frames(gif_path, max_frames=max_frames, max_side=max_side)
                if not frames:
                    scenes.append("A short clip showing a simple situation.")
                    n1s.append("object")
                    n2s.append("person")
                    continue

                scene, (a1, a2) = self.extract_scene_and_anchors(frames)
                scenes.append(scene)
                n1s.append(a1)
                n2s.append(a2)

                if (i + 1) % 25 == 0 or (i + 1) == n:
                    print(f"vl_scene_extractor: {i + 1}/{n}")
        finally:
            if unload_after:
                self.close()

        df_out = df.copy()
        df_out[s.scene_col] = scenes
        df_out[s.noun1_col] = n1s
        df_out[s.noun2_col] = n2s

        # reorder: put key columns first
        front: List[str] = [id_col]
        if gif_url_col:
            front.append(gif_url_col)
        if gif_path_col:
            front.append(gif_path_col)
        if prompt_text_col:
            front.append(prompt_text_col)
        front += [s.scene_col, s.noun1_col, s.noun2_col]

        cols = list(df_out.columns)
        front = [c for c in front if c in cols]
        rest = [c for c in cols if c not in front]
        df_out = df_out[front + rest]

        df_out.to_csv(out, index=False)
        print("Saved:", out)
        return out


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=6)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--unload_after", action="store_true")
    args = parser.parse_args(argv)

    cfg = PromptBuilderConfig()
    extractor = VLSceneExtractor(config=cfg)
    extractor.extract_tsv_to_csv(
        input_tsv=Path(args.input_tsv),
        output_csv=Path(args.output_csv),
        max_frames=int(args.max_frames),
        max_side=int(args.max_side),
        unload_after=bool(args.unload_after),
    )


if __name__ == "__main__":
    main()
