from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


_WS = re.compile(r"\s+")


def _one_line(s: str) -> str:
    return _WS.sub(" ", (s or "").replace("\n", " ").replace("\r", " ")).strip()


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    if not (s.startswith("{") and s.endswith("}")):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _scene_ok(scene: Any) -> bool:
    if not isinstance(scene, str):
        return False
    t = _one_line(scene)
    return 10 <= len(t) <= 220


def _anchors_ok(a: Any) -> bool:
    if not isinstance(a, list) or len(a) != 2:
        return False
    w1 = a[0] if len(a) > 0 else ""
    w2 = a[1] if len(a) > 1 else ""
    if not isinstance(w1, str) or not isinstance(w2, str):
        return False
    w1 = w1.strip().lower()
    w2 = w2.strip().lower()
    if w1 == w2:
        return False
    for w in (w1, w2):
        if len(w) < 3:
            return False
        if not re.fullmatch(r"[a-z]+", w):
            return False
    return True


@dataclass
class VLSceneExtractor:
    """
    Uses Qwen2.5-VL to extract:
      - scene: neutral one-sentence visual description
      - anchors: two lowercase single-word nouns

    IMPORTANT: no humor generation here. This is strictly "vision -> structured fields".
    """
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    attn_implementation: str = "sdpa"

    _processor: Optional[Any] = None
    _model: Optional[Any] = None

    def _load(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        if self._model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=dtype if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation=self.attn_implementation,
            )
            self._model.eval()
            try:
                self._model.generation_config.pad_token_id = self._processor.tokenizer.pad_token_id
            except Exception:
                pass

        return self._model, self._processor

    def _gen(
        self,
        frames: List[Image.Image],
        system: str,
        user: str,
        *,
        max_new_tokens: int = 180,
        temperature: float = 0.4,
        top_p: float = 0.95,
    ) -> str:
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
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
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
        max_tries: int = 6,
    ) -> Tuple[str, Tuple[str, str]]:
        SYSTEM = (
            "Return ONLY valid JSON. No markdown. No extra text.\n"
            "Schema:\n"
            '{"scene":"<neutral description>","anchors":["<noun1>","<noun2>"]}\n'
            "Rules:\n"
            "- scene: neutral one-sentence description of visible content.\n"
            "- anchors: exactly two single-word common nouns, lowercase letters a-z.\n"
        )
        USER = "Analyze the images and produce the JSON."

        raw = self._gen(frames, SYSTEM, USER)
        obj = _try_parse_json(raw)

        tries = 1
        while tries < max_tries:
            if obj and _scene_ok(obj.get("scene")) and _anchors_ok(obj.get("anchors")):
                break
            fix_user = (
                "Your previous output did not match the schema or rules.\n"
                "Fix it and return ONLY valid JSON.\n"
                f"Previous:\n{raw}\n"
            )
            raw = self._gen(frames, SYSTEM, fix_user, temperature=0.25, top_p=0.9)
            obj = _try_parse_json(raw)
            tries += 1

        if not obj or not _scene_ok(obj.get("scene")) or not _anchors_ok(obj.get("anchors")):
            return ("A short clip showing a simple situation.", ("object", "person"))

        scene = _one_line(obj["scene"])
        a1, a2 = obj["anchors"][0].strip().lower(), obj["anchors"][1].strip().lower()
        return (scene, (a1, a2))
