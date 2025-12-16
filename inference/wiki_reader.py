from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Iterable, List, Tuple

import requests


@dataclass
class WikipediaReader:
    """
    - title sanitization (safe_title)
    - one-line normalization
    - disambiguation detection + wipeout
    - variants for better hit rate
    - summary endpoint first, then search fallback
    - choose best extract by longest non-disambiguation candidate
    - cache with "store miss as empty string"
    - save_cache trims and wipes disambiguation-like extracts
    """

    summary_url_template: str = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    search_url: str = "https://en.wikipedia.org/w/api.php"


    headers: Dict[str, str] = field(
        default_factory=lambda: {
            # Wikimedia requires a descriptive User-Agent.
            "User-Agent": "MWAHAHA/1.0 (contact: dardemtum@gmail.com) humor-generation"
        }
    )

    timeout_seconds: float = 12.0
    sleep_seconds: float = 0.10  # set to 0.10 for politeness if you do many misses
    search_limit: int = 6
    cache_max_entries: int = 50_000  # soft cap for your own use

    # Local cache storage (previously passed around explicitly)
    cache: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def normalize_one_line(text: str) -> str:
        text = "" if text is None else str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def safe_title(cls, text: str) -> str:
        """
        Conservative sanitizer for Wikipedia titles:
        - keeps letters, digits, spaces, hyphens, apostrophes
        - drops other punctuation
        """
        text = cls.normalize_one_line(text)
        if not text:
            return ""
        text = re.sub(r"[^A-Za-z0-9 \-']", "", text).strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def looks_like_disambiguation(cls, extract: str, page_type: str | None) -> bool:
        """
        Wikipedia disambiguation pages often produce extracts like:
        - "X may refer to:"
        - "X commonly refers to:"
        """
        t = cls.normalize_one_line(extract).lower()
        if not t:
            return False
        if page_type and cls.normalize_one_line(page_type).lower() == "disambiguation":
            return True
        if "may refer to" in t:
            return True
        if "commonly refers to" in t:
            return True
        return False

    @classmethod
    def wipe_disambiguation_extract(cls, extract: str) -> str:
        """
        If the extract is disambiguation-like, return empty string.
        Otherwise return the normalized extract.
        """
        t = cls.normalize_one_line(extract)
        if not t:
            return ""
        low = t.lower()
        if "may refer to" in low or "commonly refers to" in low:
            return ""
        return t

    @classmethod
    def word_variants(cls, word: str) -> List[str]:
        """
        Generate variants to improve hit rate:
        - casing variants
        - basic plural reduction
        """
        w = cls.safe_title(word)
        if not w:
            return []

        variants = [w]
        wl = w.lower()

        # casing
        variants.append(w.lower())
        variants.append(w.title())

        # plural and inflection heuristics
        # bullies -> bully
        if wl.endswith("ies") and len(wl) > 4:
            variants.append(w[:-3] + "y")

        # watches -> watch, boxes -> box
        if wl.endswith("es") and len(wl) > 4:
            variants.append(w[:-2])

        # astronauts -> astronaut
        if wl.endswith("s") and len(wl) > 3 and not wl.endswith("ss"):
            variants.append(w[:-1])

        # deduplicate case-insensitive
        seen = set()
        out: List[str] = []
        for v in variants:
            v2 = cls.safe_title(v)
            if not v2:
                continue
            key = v2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(v2)

        return out

    @classmethod
    def pick_best_extract_from_candidates(
        cls, candidates: Iterable[Tuple[str, str, str]]
    ) -> str:
        """
        candidates yields tuples: (extract, page_type, title_used)

        Strategy:
        - ignore empty
        - ignore disambiguation-like
        - prefer longer extracts
        """
        best = ""
        best_len = -1

        for extract, page_type, _title in candidates:
            extract = cls.normalize_one_line(extract)
            if not extract:
                continue
            if cls.looks_like_disambiguation(extract, page_type):
                continue
            if len(extract) > best_len:
                best = extract
                best_len = len(extract)

        return best

    # ---------------------------------------------------------------------
    # Network calls (same endpoints + behavior)
    # ---------------------------------------------------------------------
    def fetch_summary(self, title: str) -> Tuple[str, str]:
        """
        Returns (extract, page_type). Empty strings on failure.
        """
        t = self.safe_title(title)
        if not t:
            return ("", "")

        url = self.summary_url_template.format(requests.utils.quote(t))
        try:
            r = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
            if r.status_code != 200:
                return ("", "")
            data = r.json()
            extract = self.normalize_one_line(data.get("extract", "") or "")
            page_type = self.normalize_one_line(data.get("type", "") or "")
            return (extract, page_type)
        except Exception:
            return ("", "")

    def search_titles(self, query: str) -> List[str]:
        """
        Uses the Wikipedia search endpoint to find candidate page titles.
        """
        q = self.safe_title(query)
        if not q:
            return []

        params = {
            "action": "query",
            "list": "search",
            "srsearch": q,
            "format": "json",
            "utf8": 1,
            "srlimit": int(self.search_limit),
        }

        try:
            r = requests.get(
                self.search_url,
                params=params,
                headers=self.headers,
                timeout=self.timeout_seconds,
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
    # Main entry points (same logic as get_wikipedia_extract + format block)
    # ---------------------------------------------------------------------
    def get_wikipedia_extract(self, word: str) -> str:
        """
        Main entry point for inference.

        Uses:
        - local cache
        - summary endpoint with variants
        - search fallback when needed
        - disambiguation wipeout
        """
        key = self.safe_title(word)
        if not key:
            return ""

        # Cache hit
        cached = self.cache.get(key)
        if cached is not None:
            return self.wipe_disambiguation_extract(cached)

        # Optional pacing for many misses
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        # 1) direct summary tries using variants
        direct_candidates: List[Tuple[str, str, str]] = []
        for cand in self.word_variants(key):
            extract, page_type = self.fetch_summary(cand)
            direct_candidates.append((extract, page_type, cand))

        best = self.pick_best_extract_from_candidates(direct_candidates)
        if best:
            self.cache[key] = best
            return best

        # 2) search fallback: resolve ambiguity or missing title
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

    def format_facts_block(self, word1: str, word2: str) -> str:
        """
        Builds a compact facts block for prompting.

        Keeps it short because summary extracts can be long.
        """
        w1 = self.safe_title(word1)
        w2 = self.safe_title(word2)

        e1 = self.get_wikipedia_extract(w1) if w1 else ""
        e2 = self.get_wikipedia_extract(w2) if w2 else ""

        # Cap length to avoid bloating prompts
        def cap(s: str, n: int = 260) -> str:
            s2 = self.normalize_one_line(s)
            return s2 if len(s2) <= n else (s2[:n].rstrip() + "…")

        lines = ["FACTS:"]
        lines.append(f"- {w1}: {cap(e1)}" if w1 else "- :")
        lines.append(f"- {w2}: {cap(e2)}" if w2 else "- :")
        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Cache load/save (same behavior as before)
    # ---------------------------------------------------------------------
    def load_cache(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            self.cache = {}
            return self.cache

        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Cache file is not a dictionary: {path}")

        # Normalize keys and values
        out: Dict[str, str] = {}
        for k, v in obj.items():
            kk = self.safe_title(str(k))
            if not kk:
                continue
            out[kk] = self.normalize_one_line(str(v)) if v is not None else ""

        self.cache = out
        return self.cache

    def save_cache(self, path: Path) -> None:
        """
        Saves a trimmed cache, dropping disambiguation-like extracts.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        items = list(self.cache.items())

        if len(items) > self.cache_max_entries:
            items = items[-self.cache_max_entries :]

        cleaned: Dict[str, str] = {}
        for k, v in items:
            v2 = self.wipe_disambiguation_extract(v)
            cleaned[k] = v2

        path.write_text(
            json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
        )
