from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

from config import WikiSettings


@dataclass
class WikipediaReader:
    """Wikipedia reader with all settings injected via WikiSettings."""

    settings: WikiSettings

    # Local cache storage
    cache: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._ws_re = re.compile(self.settings.whitespace_pattern)

    # ---------------------------------------------------------------------
    # Normalization / sanitization
    # ---------------------------------------------------------------------
    def normalize_one_line(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = self._ws_re.sub(" ", text).strip()
        return text

    def safe_title(self, text: str) -> str:
        t = self.normalize_one_line(text)
        if not t:
            return ""
        if self.settings.normalize_curly_apostrophe:
            t = t.replace("’", "'")
        t = re.sub(self.settings.title_allowed_chars_pattern, "", t).strip()
        t = self._ws_re.sub(" ", t).strip()
        return t

    # ---------------------------------------------------------------------
    # Disambiguation logic
    # ---------------------------------------------------------------------
    def looks_like_disambiguation(self, extract: str, page_type: str | None) -> bool:
        t = self.normalize_one_line(extract).lower()
        if not t:
            return False

        if page_type and self.normalize_one_line(page_type).lower() == self.settings.disambiguation_page_type:
            return True

        for phrase in self.settings.disambiguation_phrases:
            if phrase in t:
                return True

        return False

    def wipe_disambiguation_extract(self, extract: str) -> str:
        t = self.normalize_one_line(extract)
        if not t:
            return ""
        low = t.lower()
        for phrase in self.settings.disambiguation_phrases:
            if phrase in low:
                return ""
        return t

    # ---------------------------------------------------------------------
    # Variants
    # ---------------------------------------------------------------------
    def word_variants(self, word: str) -> List[str]:
        w = self.safe_title(word)
        if not w:
            return []

        variants: List[str] = [w]
        wl = w.lower()

        if self.settings.variant_add_lower:
            variants.append(w.lower())
        if self.settings.variant_add_title:
            variants.append(w.title())

        minlen = int(self.settings.variant_min_len_for_plural_rules)

        if self.settings.variant_ies_to_y and wl.endswith("ies") and len(wl) > minlen:
            variants.append(w[:-3] + "y")

        if self.settings.variant_strip_plural_es and wl.endswith("es") and len(wl) > minlen:
            variants.append(w[:-2])

        if self.settings.variant_strip_plural_s and wl.endswith("s") and len(wl) > minlen and not wl.endswith("ss"):
            variants.append(w[:-1])

        seen = set()
        out: List[str] = []
        for v in variants:
            v2 = self.safe_title(v)
            if not v2:
                continue
            key = v2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(v2)

        return out

    # ---------------------------------------------------------------------
    # Selection strategy
    # ---------------------------------------------------------------------
    def pick_best_extract_from_candidates(self, candidates: Iterable[Tuple[str, str, str]]) -> str:
        best = ""
        best_len = -1

        for extract, page_type, _title in candidates:
            extract = self.normalize_one_line(extract)
            if not extract:
                continue
            if self.looks_like_disambiguation(extract, page_type):
                continue
            if len(extract) > best_len:
                best = extract
                best_len = len(extract)

        return best

    # ---------------------------------------------------------------------
    # Network calls
    # ---------------------------------------------------------------------
    def fetch_summary(self, title: str) -> Tuple[str, str]:
        t = self.safe_title(title)
        if not t:
            return ("", "")

        url = self.settings.summary_url_template.format(requests.utils.quote(t))
        try:
            r = requests.get(url, headers=self.settings.headers, timeout=self.settings.timeout_seconds)
            if r.status_code != 200:
                return ("", "")
            data = r.json()
            extract = self.normalize_one_line(data.get("extract", "") or "")
            page_type = self.normalize_one_line(data.get("type", "") or "")
            return (extract, page_type)
        except Exception:
            return ("", "")

    def search_titles(self, query: str) -> List[str]:
        q = self.safe_title(query)
        if not q:
            return []

        params = {
            "action": "query",
            "list": "search",
            "srsearch": q,
            "format": "json",
            "utf8": 1,
            "srlimit": int(self.settings.search_limit),
        }

        try:
            r = requests.get(
                self.settings.search_url,
                params=params,
                headers=self.settings.headers,
                timeout=self.settings.timeout_seconds,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            items = (data.get("query", {}) or {}).get("search", []) or []
            titles: List[str] = []
            for it in items:
                title = self.normalize_one_line(it.get("title", "") or "")
                if title:
                    titles.append(title)
            return titles
        except Exception:
            return []

    # ---------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------
    def get_wikipedia_extract(self, word: str) -> str:
        key = self.safe_title(word)
        if not key:
            return ""

        cached = self.cache.get(key)
        if cached is not None:
            return self.wipe_disambiguation_extract(cached)

        if self.settings.sleep_seconds > 0:
            time.sleep(self.settings.sleep_seconds)

        direct_candidates: List[Tuple[str, str, str]] = []
        for cand in self.word_variants(key):
            extract, page_type = self.fetch_summary(cand)
            direct_candidates.append((extract, page_type, cand))

        best = self.pick_best_extract_from_candidates(direct_candidates)
        if best:
            self.cache[key] = best
            return best

        search_candidates: List[Tuple[str, str, str]] = []
        for q in self.word_variants(key):
            for title in self.search_titles(q):
                if title.lower().endswith("(disambiguation)"):
                    continue
                extract, page_type = self.fetch_summary(title)
                search_candidates.append((extract, page_type, title))

        best = self.pick_best_extract_from_candidates(search_candidates)
        if best:
            self.cache[key] = best
            return best

        self.cache[key] = ""
        return ""

    # ---------------------------------------------------------------------
    # Cache load/save
    # ---------------------------------------------------------------------
    def load_cache(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            self.cache = {}
            return self.cache

        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Cache file is not a dictionary: {path}")

        out: Dict[str, str] = {}
        for k, v in obj.items():
            kk = self.safe_title(str(k))
            if not kk:
                continue
            out[kk] = self.normalize_one_line(str(v)) if v is not None else ""

        self.cache = out
        return self.cache

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        items = list(self.cache.items())
        if len(items) > self.settings.cache_max_entries:
            items = items[-self.settings.cache_max_entries :]

        cleaned: Dict[str, str] = {}
        for k, v in items:
            cleaned[k] = self.wipe_disambiguation_extract(v)

        path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
