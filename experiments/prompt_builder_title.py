"""
Prompt builder for Task A (News Headline).

Design:
- Two-pass generation:
  1) Plan: connect headline + two anchor nouns into a joke plan
  2) Final: produce exactly one-line joke under 30 words
- Micro-cards: Wikipedia summary extract for each anchor noun (compact FACTS block)

Dependencies:
- requests
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

import requests


_whitespace_re = re.compile(r"\s+")


def normalize_one_line(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def safe_word(word: str) -> str:
    return normalize_one_line(word).strip()


WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


@lru_cache(maxsize=2048)
def get_wikipedia_extract_cached(word: str) -> str:
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
    w = safe_word(word)
    e = normalize_one_line(extract)

    if not e:
        return {"word": w, "what": "", "domain": "", "keywords": []}

    first = re.split(r"(?<=[.!?])\s+", e, maxsplit=1)[0]
    what = normalize_one_line(first)
    if len(what) > 160:
        what = what[:157].rstrip() + "..."

    domain = _guess_domain(e)

    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", e)
    tokens_lower = [t.lower() for t in tokens]
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
        "to", "of", "in", "on", "for", "at", "by", "with", "from", "as",
        "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those",
    }
    keywords: List[str] = []
    for t in tokens_lower:
        if len(t) >= 5 and t not in stop and t not in keywords:
            keywords.append(t)
        if len(keywords) >= 6:
            break

    return {"word": w, "what": what, "domain": domain, "keywords": keywords}


def build_microcards_dict(noun1: str, noun2: str) -> Dict[str, Dict[str, object]]:
    n1 = safe_word(noun1)
    n2 = safe_word(noun2)

    e1 = get_wikipedia_extract_cached(n1)
    e2 = get_wikipedia_extract_cached(n2)

    c1 = microcard_from_extract(n1, e1)
    c2 = microcard_from_extract(n2, e2)

    return {
        n1: {"what": c1.get("what", ""), "domain": c1.get("domain", ""), "keywords": c1.get("keywords", [])},
        n2: {"what": c2.get("what", ""), "domain": c2.get("domain", ""), "keywords": c2.get("keywords", [])},
    }


def format_facts_block(noun1: str, noun2: str) -> str:
    microcards = build_microcards_dict(noun1, noun2)
    n1 = safe_word(noun1)
    n2 = safe_word(noun2)

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

    return "FACTS:\n" + fmt_one(n1) + "\n" + fmt_one(n2)


def build_plan_messages(headline: str, noun1: str, noun2: str) -> List[Dict[str, str]]:
    h = normalize_one_line(headline)
    n1 = safe_word(noun1)
    n2 = safe_word(noun2)

    system = (
        "You are a stand-up comedian. First write a short private plan for a joke.\n"
        "Do not write the joke yet. Output only the plan.\n"
        "Keep it concise and practical. No preface, no emojis.\n"
        "You may receive a FACTS block; use it only to understand the anchor nouns and do not quote it."
    )

    facts = format_facts_block(n1, n2)

    user = (
        f"{facts}\n\n"
        f"Headline: {h}\n"
        f"Anchor nouns: '{n1}', '{n2}'.\n\n"
        "Task: Create a short plan for a one-line joke inspired by the headline.\n"
        "The final joke must include BOTH anchor nouns exactly as written.\n"
        "Output format: a single-line JSON object with keys "
        "\"angle\", \"misdirection\", \"word_placement\", \"device\".\n"
        "Do not include the final joke."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_final_messages(headline: str, noun1: str, noun2: str, plan_text: str) -> List[Dict[str, str]]:
    h = normalize_one_line(headline)
    n1 = safe_word(noun1)
    n2 = safe_word(noun2)
    plan = normalize_one_line(plan_text)

    system = (
        "You are a stand-up comedian. Write ONE original joke in English.\n"
        "Return exactly one line under 30 words. No preface, no explanation, no emojis.\n"
        "Avoid hate, slurs, explicit sex, and graphic violence.\n"
        "You may receive FACTS and PLAN blocks; use them only to guide the joke. Do not quote them."
    )

    facts = format_facts_block(n1, n2)

    user = (
        f"{facts}\n\n"
        f"Headline: {h}\n"
        f"Anchor nouns (must appear): '{n1}', '{n2}'.\n"
        f"PLAN (do not quote): {plan}\n\n"
        "Write the final joke now.\n"
        "Constraints: one line, under 30 words, include BOTH anchor nouns exactly as written.\n"
        "Output ONLY the joke."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
