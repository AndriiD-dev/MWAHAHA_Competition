"""
micro-card context + prompt builder for a joke model.

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
from typing import Dict, List

import requests


# -----------------------------
# Basic text helpers
# -----------------------------

_whitespace_re = re.compile(r"\s+")


def normalize_one_line(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def safe_word(word: str) -> str:
    return normalize_one_line(word).strip()


# -----------------------------
# Wikipedia fetch + cache
# -----------------------------

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


@lru_cache(maxsize=2048)
def get_wikipedia_extract_cached(word: str) -> str:
    """
    Fetch the Wikipedia summary extract for a given word.
    Uses an in-memory LRU cache to avoid repeated HTTP calls.
    """
    w = safe_word(word)
    if not w:
        return ""

    url = WIKI_SUMMARY_URL.format(requests.utils.quote(w))
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return ""
        data = r.json()
        extract = data.get("extract", "") or ""
        return normalize_one_line(extract)
    except Exception:
        return ""


# -----------------------------
# Micro-card extraction
# -----------------------------

_DOMAIN_HINTS = {
    "biology": ["species", "genus", "family", "organism", "plant", "animal", "fungus", "bacteria"],
    "technology": ["software", "hardware", "computer", "system", "device", "network", "protocol", "algorithm"],
    "science": ["physics", "chemistry", "mathematics", "astronomy", "geology", "theory", "experiment"],
    "geography": ["city", "country", "region", "river", "mountain", "island", "capital", "province"],
    "arts": ["film", "music", "band", "album", "novel", "painting", "artist", "composer"],
    "sports": ["team", "league", "season", "tournament", "player", "coach", "championship"],
}


def _guess_domain(extract: str) -> str:
    t = extract.lower()
    for domain, hints in _DOMAIN_HINTS.items():
        if any(h in t for h in hints):
            return domain
    return ""


def microcard_from_extract(word: str, extract: str) -> Dict[str, object]:
    """
    Convert a Wikipedia extract into a tiny micro-card:
      WHAT: the first sentence (or a shortened clause)
      DOMAIN: heuristic label (optional)
      KEYWORDS: a few salient words from the extract
    """
    w = safe_word(word)
    e = normalize_one_line(extract)

    if not e:
        return {"word": w, "what": "", "domain": "", "keywords": []}

    # WHAT: first sentence-ish
    what = e
    # Split on sentence end, but keep it short
    m = re.split(r"(?<=[.!?])\s+", e, maxsplit=1)
    if m and m[0]:
        what = m[0]
    what = normalize_one_line(what)
    if len(what) > 160:
        what = what[:157].rstrip() + "..."

    domain = _guess_domain(e)

    # KEYWORDS: simple heuristic (capitalized words or longer tokens)
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", e)
    tokens_lower = [t.lower() for t in tokens]
    # filter boring words
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
        "to", "of", "in", "on", "for", "at", "by", "with", "from", "as",
        "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those",
    }
    keywords = []
    for t in tokens_lower:
        if len(t) >= 5 and t not in stop and t not in keywords:
            keywords.append(t)
        if len(keywords) >= 6:
            break

    return {"word": w, "what": what, "domain": domain, "keywords": keywords}


def build_microcards_dict(word1: str, word2: str) -> Dict[str, Dict[str, object]]:
    """
    Returns the micro-cards as a dictionary:
      { word1: {what, domain, keywords}, word2: {what, domain, keywords} }
    """
    w1 = safe_word(word1)
    w2 = safe_word(word2)

    e1 = get_wikipedia_extract_cached(w1)
    e2 = get_wikipedia_extract_cached(w2)

    c1 = microcard_from_extract(w1, e1)
    c2 = microcard_from_extract(w2, e2)

    return {
        w1: {"what": c1.get("what", ""), "domain": c1.get("domain", ""), "keywords": c1.get("keywords", [])},
        w2: {"what": c2.get("what", ""), "domain": c2.get("domain", ""), "keywords": c2.get("keywords", [])},
    }


def format_facts_block(word1: str, word2: str) -> str:
    """
    Returns a compact FACTS block suitable for injection into the prompt.
    """
    microcards = build_microcards_dict(word1, word2)
    w1 = safe_word(word1)
    w2 = safe_word(word2)

    def fmt_one(w: str) -> str:
        card = microcards.get(w, {})
        what = normalize_one_line(card.get("what", ""))
        domain = normalize_one_line(card.get("domain", ""))
        keywords = card.get("keywords", []) or []
        kw = ", ".join(keywords[:6])
        parts = []
        if what:
            parts.append(f"WHAT: {what}")
        if domain:
            parts.append(f"DOMAIN: {domain}")
        if kw:
            parts.append(f"KEYWORDS: {kw}")
        body = " | ".join(parts) if parts else ""
        return f"- {w}: {body}".rstrip()

    facts = "FACTS:\n" + fmt_one(w1) + "\n" + fmt_one(w2)
    return facts


# -----------------------------
# Prompt builders
# -----------------------------

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
        "Write one joke that MUST include the two required words exactly as written.\n"
        f"Required words: '{w1}' and '{w2}'. "
        "Use both words naturally. One line only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_plan_messages(word1: str, word2: str) -> List[Dict[str, str]]:
    """
    Returns chat messages (system + user) for the planning step.

    The model must output ONLY a short plan (no joke yet), ideally as a single-line JSON object.
    """
    w1 = safe_word(word1)
    w2 = safe_word(word2)

    system = (
        "You are a stand-up comedian. You will first write a short private plan for a joke.\n"
        "Do NOT write the joke yet. Output only the plan.\n"
        "Keep the plan concise and practical. No preface, no emojis.\n"
        "You may receive a FACTS block; use it only to understand the words and do not quote it."
    )

    facts = format_facts_block(w1, w2)

    user = (
        f"{facts}\n\n"
        "Task: Create a short plan for a one-line joke that uses BOTH required words naturally.\n"
        f"Required words: '{w1}' and '{w2}'.\n"
        "Output format: a single-line JSON object with keys "
        "\"scenario\", \"misdirection\", \"word_placement\", \"device\".\n"
        "Do not include the final joke."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_final_messages(word1: str, word2: str, plan_text: str) -> List[Dict[str, str]]:
    """
    Returns chat messages (system + user) for the final joke step, conditioned on a plan.

    The model must output ONLY the final one-line joke (under 30 words) and must include BOTH words.
    """
    w1 = safe_word(word1)
    w2 = safe_word(word2)
    plan = normalize_one_line(plan_text)

    system = (
        "You are a stand-up comedian. Write ONE original joke in English.\n"
        "Return exactly one line under 30 words. No preface, no explanation, no emojis.\n"
        "Avoid hate, slurs, explicit sex, and graphic violence.\n"
        "You may receive FACTS and PLAN blocks; use them only to understand the words and the intended angle. "
        "Do not quote them."
    )

    facts = format_facts_block(w1, w2)

    user = (
        f"{facts}\n\n"
        f"PLAN (do not quote): {plan}\n\n"
        "Write the final joke now.\n"
        f"Constraints: must include '{w1}' and '{w2}' exactly as written.\n"
        "Output ONLY the joke."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    w1, w2 = "submarine", "cactus"

    microcards = build_microcards_dict(w1, w2)
    print("MICROCARDS:\n", microcards, "\n")

    plan_messages = build_plan_messages(w1, w2)
    print("PLAN SYSTEM MESSAGE:\n", plan_messages[0]["content"], "\n")
    print("PLAN USER MESSAGE:\n", plan_messages[1]["content"], "\n")

    final_messages = build_final_messages(w1, w2, '{"scenario":"museum","misdirection":"literal","word_placement":"punchline","device":"reversal"}')
    print("FINAL SYSTEM MESSAGE:\n", final_messages[0]["content"], "\n")
    print("FINAL USER MESSAGE:\n", final_messages[1]["content"])
