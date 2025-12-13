"""
Copy-paste-ready micro-card context + prompt builder for a joke model.

What it does:
- Fetches Wikipedia summary "extract" for each input word
- Converts extract into a tiny "micro-card" (WHAT + optional DOMAIN + KEYWORDS)
- Builds a memory-efficient prompt that injects these micro-cards as a FACTS block

Dependencies:
- requests  (pip install requests)

Notes:
- This file uses an in-memory least-recently-used cache to avoid repeated network calls.
- The prompt is intentionally short and structured to reduce token usage.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import quote

import requests

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Small stopword list (fast and dependency-free). Expand if you like.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by", "at", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "which", "who", "whom", "whose", "also", "may", "can", "often", "usually", "most", "many",
    "has", "have", "had", "into", "over", "under", "between", "during", "after", "before",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_WORD_RE = re.compile(r"[^a-zA-Z0-9\- ]+")
_PARENS_RE = re.compile(r"\([^)]*\)")


def safe_word(word: str) -> str:
    """Basic cleanup for a word or short phrase."""
    if word is None:
        return ""
    w = str(word).strip()
    # Collapse whitespace
    w = re.sub(r"\s+", " ", w)
    return w


def normalize_one_line(text: str) -> str:
    """Normalize text to a single line with collapsed whitespace."""
    if not text:
        return ""
    t = str(text).replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _first_sentence(text: str, max_chars: int = 220) -> str:
    """
    Take the first sentence and cap it for token efficiency.
    Also removes parenthetical details that are often not useful for jokes.
    """
    text = normalize_one_line(text)
    if not text:
        return ""

    # Remove parentheses like dates or side notes for compactness
    text = _PARENS_RE.sub("", text)
    text = normalize_one_line(text)

    parts = _SENTENCE_SPLIT_RE.split(text)
    sent = parts[0].strip() if parts else text.strip()

    if len(sent) > max_chars:
        sent = sent[:max_chars].rsplit(" ", 1)[0].strip()
        if sent:
            sent += "…"
    return sent


def _infer_domain(first_sentence: str) -> str:
    """
    Heuristic: capture a short noun phrase after "is/are a/an/the".
    Example: "A submarine is a watercraft that operates underwater." -> "watercraft"
    """
    s = first_sentence
    m = re.search(r"\b(?:is|are)\s+(?:an?|the)\s+([^.,;:]+)", s, flags=re.IGNORECASE)
    if not m:
        return ""
    phrase = m.group(1).strip()

    # Stop at common clause openers
    phrase = re.split(
        r"\b(?:that|which|who|where|when|while|because|although)\b",
        phrase,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Keep it short
    words = phrase.split()
    return " ".join(words[:6]).strip()


def _keywords(text: str, k: int = 6) -> List[str]:
    """
    Cheap keyword selection:
    - Lowercase
    - Strip punctuation
    - Drop stopwords
    - Keep first unique tokens (order-based)
    """
    if not text:
        return []

    t = text.lower()
    t = _PARENS_RE.sub("", t)
    t = _NON_WORD_RE.sub(" ", t)
    tokens = [tok.strip("-") for tok in t.split() if tok.strip("-")]

    out: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        if len(tok) < 3:
            continue
        if tok.isdigit():
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= k:
            break
    return out


@lru_cache(maxsize=50_000)
def fetch_wikipedia_extract(word: str, timeout_s: float = 4.0) -> str:
    """
    Cached fetch of Wikipedia summary extract for a word/title.
    Returns empty string on not found, disambiguation, or error.
    """
    w = safe_word(word)
    if not w:
        return ""

    candidates = [
        w,
        w.replace("_", " "),
        w.replace(" ", "_"),
        (w[:1].upper() + w[1:]) if w else w,
    ]

    # Deduplicate preserving order
    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    headers = {
        # Wikimedia requests a descriptive User-Agent.
        "User-Agent": "joke-context-microcards/1.0 (local inference; no email)",
        "Accept": "application/json",
    }

    with requests.Session() as session:
        for title in candidates:
            url = WIKI_SUMMARY_URL.format(quote(title, safe=""))
            try:
                resp = session.get(url, headers=headers, timeout=timeout_s)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError):
                continue

            # Skip disambiguation pages (usually noisy)
            if data.get("type") == "disambiguation":
                continue

            extract = data.get("extract")
            if isinstance(extract, str) and extract.strip():
                return extract.strip()

    return ""


def build_microcard_from_extract(extract: str) -> Dict[str, object]:
    """Convert a Wikipedia extract into a tiny micro-card."""
    what = _first_sentence(extract)
    domain = _infer_domain(what) if what else ""
    keys = _keywords(extract, k=6)
    return {"what": what, "domain": domain, "keywords": keys}


@lru_cache(maxsize=50_000)
def get_microcard(word: str) -> Dict[str, object]:
    """
    Cached micro-card builder. Stores only compact fields, not the full extract.
    """
    extract = fetch_wikipedia_extract(word)
    if not extract:
        return {"what": "", "domain": "", "keywords": []}
    return build_microcard_from_extract(extract)


def format_facts_block(word1: str, word2: str) -> str:
    """Create a compact facts block for the two words."""
    w1 = safe_word(word1)
    w2 = safe_word(word2)

    c1 = get_microcard(w1)
    c2 = get_microcard(w2)

    def render(w: str, c: Dict[str, object]) -> str:
        what = (c.get("what") or "").strip()
        domain = (c.get("domain") or "").strip()
        keywords = c.get("keywords") or []
        if not what:
            what = "(no context found)"
        keys_str = ", ".join(list(keywords)[:6]) if keywords else "none"
        if domain:
            return f'WORD: "{w}"\nWHAT: {what}\nDOMAIN: {domain}\nKEYWORDS: {keys_str}'
        return f'WORD: "{w}"\nWHAT: {what}\nKEYWORDS: {keys_str}'

    return (
        "FACTS (for grounding only; do not quote):\n"
        + render(w1, c1)
        + "\n\n"
        + render(w2, c2)
    )


def build_joke_messages(word1: str, word2: str) -> List[Dict[str, str]]:
    """
    Returns a list of chat messages (system + user) that you can pass to your model.
    """
    w1 = safe_word(word1)
    w2 = safe_word(word2)

    system = (
        "You are a stand-up comedian. Write ONE original joke in English.\n"
        "Return exactly one line under 30 words. No preface, no explanation, no emojis.\n"
        "Avoid hate, slurs, explicit sex, and graphic violence.\n"
        "You may receive a FACTS block; use it only to understand the words and do not quote it."
    )

    facts = format_facts_block(w1, w2)

    user = (
        f"{facts}\n\n"
        "TASK:\n"
        f"Write one joke that MUST include the two words exactly as written: '{w1}' and '{w2}'. "
        "Use both words naturally. One line only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_microcards_dict(word1: str, word2: str) -> Dict[str, Dict[str, object]]:
    """
    Returns the micro-cards as a dictionary:
      { word1: {what, domain, keywords}, word2: {what, domain, keywords} }
    """
    w1 = safe_word(word1)
    w2 = safe_word(word2)
    return {
        w1: get_microcard(w1),
        w2: get_microcard(w2),
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    w1, w2 = "submarine", "cactus"

    microcards = build_microcards_dict(w1, w2)
    print("MICROCARDS:\n", microcards, "\n")

    messages = build_joke_messages(w1, w2)
    print("SYSTEM MESSAGE:\n", messages[0]["content"], "\n")
    print("USER MESSAGE:\n", messages[1]["content"])
