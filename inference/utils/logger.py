from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _short(s: Optional[str], max_chars: int) -> str:
    if not s:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _truncate_value(v: Any, max_chars: int, max_list_items: int) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        return _short(v, max_chars)
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, dict):
        return {k: _truncate_value(val, max_chars, max_list_items) for k, val in v.items()}
    if isinstance(v, list):
        trimmed = v[:max_list_items]
        return [_truncate_value(x, max_chars, max_list_items) for x in trimmed]
    return _short(str(v), max_chars)


@dataclass
class Logger:
    """
    File logger for "first sample per batch" structured logs.

    Behavior (restored):
    - Buffers JSON blocks in memory during the run.
    - Writes the whole log file once at the end (on close()).

    Optional:
    - Set flush_each=True if you want streaming writes during the run.
    """

    enabled: bool = True
    log_path: Optional[Path] = None
    max_chars: int = 900
    max_list_items: int = 12

    # If True -> write each block immediately (streaming).
    # If False -> buffer and write once at the end (default, like before).
    flush_each: bool = False

    fh: Optional[TextIO] = field(default=None, init=False, repr=False)
    _blocks: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    # Best-effort extractors (so you do not have to modify PromptBuilder right now).
    _re_facts: re.Pattern = re.compile(
        r"(?is)\bFACTS\b\s*[:\-]?\s*(.*?)(?:\n\s*\bPLAN\b|\n\s*\bTask\b|\n\s*Write the final joke\b|\Z)"
    )
    _re_microcards: re.Pattern = re.compile(
        r"(?is)\bMICROCARDS?\b\s*[:\-]?\s*(.*?)(?:\n\s*\bPLAN\b|\n\s*\bTask\b|\n\s*Write the final joke\b|\Z)"
    )
    _re_plan_in_prompt: re.Pattern = re.compile(
        r"(?is)\bPLAN\b[^\n]*[:\-]?\s*(.*?)(?:\n\s*Write the final joke\b|\n\s*Constraints\b|\Z)"
    )

    def __post_init__(self) -> None:
        if not self.enabled or self.log_path is None:
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # If streaming mode, open the file now (truncate).
        if self.flush_each:
            self.fh = self.log_path.open("w", encoding="utf-8")

        # Always record run_start as the first block (buffered or streamed).
        self._write_json_block(
            {
                "type": "run_start",
                "time_iso": _now_iso(),
                "time_hms": _now_hms(),
                "log_path": str(self.log_path),
                "flush_each": bool(self.flush_each),
            }
        )

    def close(self) -> None:
        """
        Flush and close.
        - If flush_each=True, just close the handle.
        - If flush_each=False, write buffered blocks to disk once.
        """
        if not self.enabled or self.log_path is None:
            return

        if self.flush_each:
            if self.fh is not None:
                try:
                    self.fh.flush()
                finally:
                    self.fh.close()
                self.fh = None
            return

        # Buffered mode: write everything once at the end.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            for block in self._blocks:
                txt = json.dumps(block, indent=2, ensure_ascii=False, default=str)
                f.write(txt)
                f.write("\n\n")

        self._blocks.clear()

    def _write_json_block(self, obj: Dict[str, Any]) -> None:
        if not self.enabled or self.log_path is None:
            return

        safe_obj = _truncate_value(obj, self.max_chars, self.max_list_items)

        if self.flush_each:
            if self.fh is None:
                # Should not happen, but be robust.
                self.fh = self.log_path.open("w", encoding="utf-8")
            txt = json.dumps(safe_obj, indent=2, ensure_ascii=False, default=str)
            self.fh.write(txt)
            self.fh.write("\n\n")
            self.fh.flush()
            return

        # Buffered mode: store safe blocks in memory, write at close()
        self._blocks.append(safe_obj)

    def try_parse_json(self, s: str) -> Optional[Any]:
        s = (s or "").strip()
        if not s:
            return None
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return None
        return None

    def extract_prompt_blocks(self, chat_text: str) -> Dict[str, Any]:
        txt = chat_text or ""

        facts = ""
        microcards_raw = ""
        plan_in_prompt = ""

        m = self._re_facts.search(txt)
        if m:
            facts = (m.group(1) or "").strip()

        m = self._re_microcards.search(txt)
        if m:
            microcards_raw = (m.group(1) or "").strip()

        m = self._re_plan_in_prompt.search(txt)
        if m:
            plan_in_prompt = (m.group(1) or "").strip()

        microcards: List[str] = []
        if microcards_raw:
            microcards = self._split_microcards(microcards_raw)
        elif facts:
            microcards = self._split_microcards(facts)

        return {
            "facts": facts,
            "microcards": microcards,
            "plan_in_prompt": plan_in_prompt,
        }

    def _split_microcards(self, block: str) -> List[str]:
        if not block:
            return []

        lines = [ln.strip() for ln in block.split("\n")]
        items: List[str] = []

        for ln in lines:
            if not ln:
                continue
            if ln.startswith(("-", "*", "•")):
                items.append(ln.lstrip("-*•").strip())
            elif re.match(r"^\d+[.)]\s+", ln):
                items.append(re.sub(r"^\d+[.)]\s+", "", ln).strip())

        if items:
            return [x for x in items if x][: self.max_list_items]

        paras = [p.strip() for p in block.split("\n\n") if p.strip()]
        return paras[: self.max_list_items]

    def log_run_meta(self, meta: Dict[str, Any]) -> None:
        self._write_json_block({"type": "run_meta", "time_iso": _now_iso(), "meta": meta})

    def log_first_in_batch(
        self,
        *,
        stage: str,
        batch_index: int,
        batch_size: int,
        example_index: int,
        chat_text: str,
        model_output: str,
        meta: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> None:
        blocks = self.extract_prompt_blocks(chat_text)
        parsed_output = self.try_parse_json(model_output)

        event: Dict[str, Any] = {
            "type": "sample",
            "time_iso": _now_iso(),
            "time_hms": _now_hms(),
            "stage": stage,
            "batch": {"index": batch_index, "size": batch_size},
            "example": {"index": example_index},
            "anchors": (meta or {}).get("anchors"),
            "headline": (meta or {}).get("headline"),
            "wikipedia_context": (meta or {}).get("wikipedia_context") or blocks.get("facts"),
            "microcards": (meta or {}).get("microcards") or blocks.get("microcards"),
            "plan": (meta or {}).get("plan"),
            "prompt_plan_block": blocks.get("plan_in_prompt"),
            "evaluation": evaluation,
            "prompt_preview": _short(chat_text, self.max_chars),
            "model_output_raw": _short(model_output, self.max_chars),
            "model_output_parsed": parsed_output,
        }
        if note:
            event["note"] = note

        self._write_json_block(event)

    def log_failed_jokes(self, items: List[Dict[str, Any]]) -> None:
        self._write_json_block(
            {
                "type": "failed_jokes",
                "time_iso": _now_iso(),
                "count": len(items),
                "items": items,
            }
        )

    def log_run_end(self, summary: Dict[str, Any]) -> None:
        self._write_json_block(
            {
                "type": "run_end",
                "time_iso": _now_iso(),
                "summary": summary,
            }
        )

    def log_fallback_predictions(self, items: List[Dict[str, Any]]) -> None:
        self._write_json_block(
            {
                "type": "fallback_predictions",
                "time_iso": _now_iso(),
                "count": len(items),
                "items": items,
            }
        )
