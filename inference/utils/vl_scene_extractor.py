from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageSequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import pandas as pd
import requests

from inference.config import PromptBuilderConfig


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
    Evenly sample up to max_frames from a GIF, convert to RGB, resize to max_side.
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


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for want in candidates:
        k = want.lower()
        if k in cols:
            return cols[k]
    return ""


def _count_words(s: str) -> int:
    s = _one_line(s)
    if not s:
        return 0
    return len([w for w in s.split(" ") if w])


def _valid_ctx(obj: Dict[str, Any], required_keys: Tuple[str, ...]) -> bool:
    # Strict keys: no extras, no missing.
    if set(obj.keys()) != set(required_keys):
        return False

    # Types: all must be strings.
    for k in required_keys:
        if not isinstance(obj.get(k), str):
            return False

    image_facts = _one_line(obj.get("image_facts", ""))
    if not image_facts:
        return False

    # Avoid multi-sentence dumps (keep robust; allow 1-2 punctuations).
    if (image_facts.count(".") + image_facts.count("!") + image_facts.count("?")) > 2:
        return False

    wc = _count_words(image_facts)
    if wc < 10 or wc > 30:
        return False

    # Avoid joke/intent markers
    if re.search(r"\b(joke|funny|hilarious|lol|lmao)\b", image_facts, flags=re.IGNORECASE):
        return False

    # No filler for key_objects
    key_objects = _one_line(obj.get("key_objects", "")).lower()
    if key_objects:
        if re.search(r"\b(thing|stuff|object|item|someone|anyone|everyone|person|people)\b", key_objects):
            return False

    return True


def _sanitize_ctx(obj: Dict[str, Any]) -> Dict[str, str]:
    return {k: _one_line(str(v or "")) for k, v in obj.items()}


def batched(indices: List[int], batch_size: int) -> List[List[int]]:
    out: List[List[int]] = []
    cur: List[int] = []
    for i in indices:
        cur.append(i)
        if len(cur) >= batch_size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


@dataclass
class VLSceneExtractor:
    """
    Vision-language preprocessing:
      - input: GIF frames
      - output: strict JSON -> image context only
      - TSV -> CSV: add context columns (no nouns)
      - logs: write one sample per batch into a JSONL file
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
        try:
            if self._model is not None:
                try:
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

    def extract_context(
        self,
        frames: List[Image.Image],
        *,
        max_tries: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Returns a strict dict with keys:
        image_facts, ambience, written_text, key_objects, actions, setting, camera_style
        """
        s = self.config.vl_scene_extractor
        required = tuple(s.json_required_keys)
        tries_limit = int(max_tries) if max_tries is not None else int(s.max_tries)

        system = s.system_prompt
        user = s.user_prompt

        raw = self._gen(frames, system, user)
        obj = _try_parse_json(raw)

        tries = 1
        while tries < tries_limit:
            if obj and _valid_ctx(obj, required):
                return _sanitize_ctx(obj)

            fix_user = s.repair_user_template.format(previous=raw)
            raw = self._gen(frames, system, fix_user)
            obj = _try_parse_json(raw)
            tries += 1

        # Fallback: keep schema valid (strict keys), keep it neutral.
        return {
            "image_facts": "A short clip showing a simple situation.",
            "ambience": "",
            "written_text": "",
            "key_objects": "",
            "actions": "",
            "setting": "",
            "camera_style": "",
        }

    # ----------------------------
    # TSV -> CSV preprocessing + batch logging
    # ----------------------------

    def extract_tsv_to_csv(
        self,
        *,
        input_tsv: Path,
        output_csv: Optional[Path] = None,
        max_frames: int = 6,
        max_side: int = 512,
        batch_size: int = 16,
        unload_after: bool = True,
        download_timeout_seconds: int = 60,
    ) -> Path:
        """
        Reads a tab separated values file with a GIF location column.
        Writes a comma separated values file with extracted image context fields.

        Also writes one sampled extraction per batch into:
          outputs/<stem>.vl_samples.jsonl
        """
        df = pd.read_csv(input_tsv, sep="\t", keep_default_na=False)
        s = self.config.vl_scene_extractor

        id_col = _pick_col(df, ["id"])
        gif_url_col = _pick_col(df, ["gif_url", "url"])
        gif_path_col = _pick_col(df, ["gif_path", "path", "filepath", "file"])
        prompt_text_col = _pick_col(df, ["prompt_text", "prompt", "text"])

        if not id_col:
            df.insert(0, "id", [f"row_{i:05d}" for i in range(len(df))])
            id_col = "id"

        if not gif_url_col and not gif_path_col:
            raise ValueError("Input must contain a GIF url column (gif_url/url) or a GIF path column (gif_path/path).")

        out = output_csv or (self.config.paths.data_dir / f"{Path(input_tsv).stem}.scenes.csv")
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp_dir = self.config.paths.data_dir / "_tmp_gifs"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Separate per-batch sample log (JSON Lines)
        samples_path = self.config.paths.output_dir / f"{Path(input_tsv).stem}.vl_samples.jsonl"
        samples_path.parent.mkdir(parents=True, exist_ok=True)

        scenes: List[str] = [""] * len(df)
        ambiences: List[str] = [""] * len(df)
        written_texts: List[str] = [""] * len(df)
        key_objects_list: List[str] = [""] * len(df)
        actions_list: List[str] = [""] * len(df)
        settings_list: List[str] = [""] * len(df)
        camera_styles: List[str] = [""] * len(df)

        indices = list(range(len(df)))
        batches = batched(indices, max(1, int(batch_size)))

        # append mode: keep history across runs unless you delete it
        with samples_path.open("a", encoding="utf-8") as sample_f:
            try:
                for batch_index, batch_ids in enumerate(batches):
                    t_batch0 = time.time()

                    first_sample_written = False

                    for i in batch_ids:
                        rid = str(df.loc[i, id_col]).strip() or f"row_{i:05d}"

                        local = ""
                        if gif_path_col:
                            local = str(df.loc[i, gif_path_col]).strip()
                        if not local and gif_url_col:
                            local = str(df.loc[i, gif_url_col]).strip()

                        if not local:
                            ctx = {
                                "image_facts": "A short clip showing a simple situation.",
                                "ambience": "",
                                "written_text": "",
                                "key_objects": "",
                                "actions": "",
                                "setting": "",
                                "camera_style": "",
                            }
                        else:
                            if _is_url(local):
                                gif_path = _download(local, tmp_dir / f"{rid}.gif", timeout_seconds=download_timeout_seconds)
                            else:
                                gif_path = Path(local)

                            frames = _extract_gif_frames(gif_path, max_frames=max_frames, max_side=max_side)
                            ctx = self.extract_context(frames) if frames else {
                                "image_facts": "A short clip showing a simple situation.",
                                "ambience": "",
                                "written_text": "",
                                "key_objects": "",
                                "actions": "",
                                "setting": "",
                                "camera_style": "",
                            }

                        scenes[i] = ctx["image_facts"]
                        ambiences[i] = ctx["ambience"]
                        written_texts[i] = ctx["written_text"]
                        key_objects_list[i] = ctx["key_objects"]
                        actions_list[i] = ctx["actions"]
                        settings_list[i] = ctx["setting"]
                        camera_styles[i] = ctx["camera_style"]

                        # Log exactly one sample per batch into file
                        if not first_sample_written:
                            payload = {
                                "stage": "vl_extract",
                                "batch_index": int(batch_index),
                                "batch_size": int(len(batch_ids)),
                                "example_index": int(i),
                                "id": rid,
                                "gif_source": local,
                                "context": ctx,
                            }
                            sample_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            sample_f.flush()
                            first_sample_written = True

                    t_batch1 = time.time()
                    print(f"[VISION_LANGUAGE] batch={batch_index} size={len(batch_ids)} seconds={t_batch1 - t_batch0:.2f}")

            finally:
                if unload_after:
                    self.close()

        df_out = df.copy()
        # Keep `scene` column for compatibility; it stores image_facts.
        df_out[s.scene_col] = scenes
        df_out[s.ambience_col] = ambiences
        df_out[s.written_text_col] = written_texts
        df_out[s.key_objects_col] = key_objects_list
        df_out[s.actions_col] = actions_list
        df_out[s.setting_col] = settings_list
        df_out[s.camera_style_col] = camera_styles

        # reorder: put key columns first
        front: List[str] = [id_col]
        if gif_url_col:
            front.append(gif_url_col)
        if gif_path_col:
            front.append(gif_path_col)
        if prompt_text_col:
            front.append(prompt_text_col)
        front += [
            s.scene_col,
            s.ambience_col,
            s.written_text_col,
            s.key_objects_col,
            s.actions_col,
            s.setting_col,
            s.camera_style_col,
        ]

        cols = list(df_out.columns)
        front = [c for c in front if c in cols]
        rest = [c for c in cols if c not in front]
        df_out = df_out[front + rest]

        df_out.to_csv(out, index=False)
        print("Saved:", out)
        print("Batch samples log:", samples_path)
        return out


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=6)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unload_after", action="store_true")
    args = parser.parse_args(argv)

    cfg = PromptBuilderConfig()
    extractor = VLSceneExtractor(config=cfg)
    extractor.extract_tsv_to_csv(
        input_tsv=Path(args.input_tsv),
        output_csv=Path(args.output_csv),
        max_frames=int(args.max_frames),
        max_side=int(args.max_side),
        batch_size=int(args.batch_size),
        unload_after=bool(args.unload_after),
    )


if __name__ == "__main__":
    main()
