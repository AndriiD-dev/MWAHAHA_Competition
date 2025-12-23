from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from PIL import Image


@dataclass(frozen=True)
class GifFrameSettings:
    max_frames: int = 4
    max_side: int = 512
    timeout_seconds: float = 60.0


def download_gif(url: str, out_path: Path, timeout_seconds: float = 60.0) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout_seconds)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def extract_frames(path: Path, *, max_frames: int = 4, max_side: int = 512) -> List[Image.Image]:
    im = Image.open(path)
    frames: List[Image.Image] = []
    n = getattr(im, "n_frames", 1)

    if n <= 1:
        frames.append(im.convert("RGB"))
        return frames

    step = max(1, n // max_frames)
    idxs = list(range(0, n, step))[:max_frames]

    for i in idxs:
        im.seek(i)
        fr = im.convert("RGB")

        w, h = fr.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            fr = fr.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        frames.append(fr)

    return frames
