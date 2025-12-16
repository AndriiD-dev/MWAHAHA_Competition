from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib.util


# -----------------------------------------------------------------------------
# Load your existing prompt builder (reuse formatting + final prompt structure)
# -----------------------------------------------------------------------------
PROMPT_BUILDER_PATH = Path("../inference/task_a/two_words/prompt_builder_two_words.py")

def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

pb = import_module_from_path("prompt_builder_two_words", PROMPT_BUILDER_PATH)


# -----------------------------------------------------------------------------
# spaCy noun extraction with filtering + preference for common nouns
# -----------------------------------------------------------------------------
def _load_spacy_model():
    try:
        import spacy
    except Exception as e:
        raise RuntimeError(
            "spaCy is required.\n"
            "Install and download the English model:\n"
            "pip install spacy\n"
            "python -m spacy download en_core_web_sm"
        ) from e

    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        raise RuntimeError(
            "spaCy model en_core_web_sm is missing.\n"
            "Run:\n"
            "python -m spacy download en_core_web_sm"
        ) from e

_NLP = None

GENERIC_NOUNS = {
    "thing", "things", "stuff", "something", "anything", "everything",
    "someone", "anyone", "everyone", "somebody", "anybody", "everybody",
    "person", "people", "man", "men", "woman", "women", "guy", "guys", "girl", "girls", "kid", "kids",
    "friend", "friends", "family",
    "time", "day", "week", "month", "year", "moment",
    "place", "home", "house", "room",
    "job", "work", "boss", "company",
    "way", "lot", "kind", "sort", "part", "case", "point", "problem", "idea", "fact", "question", "answer",
    "joke", "story",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "at", "by", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "our", "ours", "they", "their", "theirs",
    "he", "him", "his", "she", "her", "hers",
    "do", "does", "did", "doing",
    "not", "no", "yes",
}

@dataclass(frozen=True)
class NounCandidate:
    surface: str      # keep original casing, because prompt requires exact match
    lower: str
    is_proper: bool
    weight: float

def noun_candidates(text: str, *, min_len: int = 3, prefer_common_nouns: bool = True) -> List[NounCandidate]:
    """
    Extract noun-like candidates:
    - uses spaCy part-of-speech: NOUN and PROPN
    - drops very short tokens
    - drops stopwords and generic nouns
    - optionally prefers common nouns by down-weighting proper nouns
    """
    global _NLP
    if _NLP is None:
        _NLP = _load_spacy_model()

    t = (text or "").strip()
    if not t:
        return []

    doc = _NLP(t)
    out: List[NounCandidate] = []
    seen_lower = set()

    for tok in doc:
        if tok.pos_ not in {"NOUN", "PROPN"}:
            continue

        surface = tok.text.strip()
        if len(surface) < min_len:
            continue

        # keep it strict for reliable “exact match” constraints
        if not surface.isalpha():
            continue

        lower = surface.lower()
        if len(lower) < min_len:
            continue
        if lower in STOPWORDS:
            continue
        if lower in GENERIC_NOUNS:
            continue
        if lower in seen_lower:
            continue

        is_proper = tok.pos_ == "PROPN"
        weight = 1.0
        if prefer_common_nouns and is_proper:
            weight = 0.25

        out.append(NounCandidate(surface=surface, lower=lower, is_proper=is_proper, weight=weight))
        seen_lower.add(lower)

    return out


# -----------------------------------------------------------------------------
# Anchor sampling with your mix + similarity checks
# -----------------------------------------------------------------------------
def _base_form_for_similarity(w: str) -> str:
    x = w.lower()
    if x.endswith("ies") and len(x) > 4:
        return x[:-3] + "y"
    if x.endswith("s") and not x.endswith("ss") and len(x) > 3:
        return x[:-1]
    return x

def choose_two_anchors_with_mix(
    setup: str,
    punchline: str,
    *,
    rnd: random.Random,
    mode_probs: Tuple[float, float, float] = (0.50, 0.25, 0.25),  # (one+one, two-setup, two-punchline)
    max_resample: int = 60,
    prefer_common_nouns: bool = True,
) -> Optional[Tuple[str, str]]:
    setup_cands = noun_candidates(setup, prefer_common_nouns=prefer_common_nouns)
    punch_cands = noun_candidates(punchline, prefer_common_nouns=prefer_common_nouns)

    if not setup_cands and not punch_cands:
        return None

    def pick_one(cands: List[NounCandidate]) -> Optional[str]:
        if not cands:
            return None
        return rnd.choices([c.surface for c in cands], weights=[c.weight for c in cands], k=1)[0]

    for _ in range(max_resample):
        r = rnd.random()

        if r < mode_probs[0]:
            a = pick_one(setup_cands) or pick_one(punch_cands)
            b = pick_one(punch_cands) or pick_one(setup_cands)
        elif r < mode_probs[0] + mode_probs[1]:
            a = pick_one(setup_cands) or pick_one(punch_cands)
            b = pick_one(setup_cands) or pick_one(punch_cands)
        else:
            a = pick_one(punch_cands) or pick_one(setup_cands)
            b = pick_one(punch_cands) or pick_one(setup_cands)

        if not a or not b:
            continue
        if a == b:
            continue
        if _base_form_for_similarity(a) == _base_form_for_similarity(b):
            continue

        return (a, b)

    return None


# -----------------------------------------------------------------------------
# Prompt message construction (reuse prompt builder, optional topic injection)
# -----------------------------------------------------------------------------
DEFAULT_PLAN = (
    '{"scenario":"everyday situation","misdirection":"literal reading",'
    '"word_placement":"punchline","device":"reversal"}'
)

def inject_topic_line(user_content: str, topic_line: str) -> str:
    topic_line = pb.normalize_one_line(topic_line).strip()
    if not topic_line:
        return user_content

    marker = "\nTask:"
    idx = user_content.find(marker)
    if idx == -1:
        return pb.normalize_one_line(user_content + "\nTOPIC: " + topic_line)

    # keep formatting stable
    return user_content[:idx] + f"\nTOPIC: {topic_line}\n" + user_content[idx+1:]


def strip_facts_block(user_content: str) -> str:
    """
    Remove FACTS block if you want fully offline dataset building.
    The prompt builder usually creates:
    FACTS: ...
    PLAN ...
    """
    # remove everything from "FACTS:" up to "PLAN"
    m = re.search(r"(?s)\AFACTS:\s*.*?\nPLAN", user_content)
    if not m:
        return user_content
    plan_idx = user_content.find("PLAN")
    if plan_idx == -1:
        return user_content
    return user_content[plan_idx:]


def build_final_messages(
    word1: str,
    word2: str,
    *,
    plan_text: str,
    topic_line: str,
    include_noun_cards: bool,
) -> List[Dict[str, str]]:
    messages = pb.build_final_messages(word1, word2, plan_text)

    # ensure we do not mutate original
    messages = [dict(messages[0]), dict(messages[1])]

    if not include_noun_cards:
        messages[1]["content"] = strip_facts_block(messages[1]["content"])

    if topic_line:
        messages[1]["content"] = inject_topic_line(messages[1]["content"], topic_line)

    return messages


# -----------------------------------------------------------------------------
# Target creation + validation
# -----------------------------------------------------------------------------
def make_one_line_target(setup: str, punchline: str) -> str:
    return pb.normalize_one_line(f"{setup} {punchline}").strip()

def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])

def contains_exact(text: str, word: str) -> bool:
    w = re.escape(word)
    return re.search(rf"(?<![A-Za-z0-9]){w}(?![A-Za-z0-9])", text) is not None


# -----------------------------------------------------------------------------
# Load input jokes
# Supports either:
# - {"setup": ..., "punchline": ...}
# - {"prompt": ..., "response": ...}  (your merged file)
# -----------------------------------------------------------------------------
def load_jokes_jsonl(path: Path) -> List[Tuple[str, str]]:
    jokes: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            setup = obj.get("setup")
            punchline = obj.get("punchline")
            if setup is None and punchline is None:
                setup = obj.get("prompt", "")
                punchline = obj.get("response", "")

            setup = pb.normalize_one_line(str(setup or "")).strip()
            punchline = pb.normalize_one_line(str(punchline or "")).strip()

            if setup and punchline:
                jokes.append((setup, punchline))
    return jokes


# -----------------------------------------------------------------------------
# Build Dataset 1: messages + target only (no rendered prompt)
# Duplicate mitigation:
# - per-joke: do not repeat the same unordered anchor pair
# - global: do not repeat (target + unordered anchor pair)
# -----------------------------------------------------------------------------
def build_dataset_1_word_inclusion_messages_only(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    samples_per_joke: int = 2,
    seed: int = 1337,
    include_noun_cards: bool = True,
    include_topic_line: bool = True,
    prefer_common_nouns: bool = True,
    max_words: int = 30,
    max_pair_attempts_per_sample: int = 80,
) -> None:
    jokes = load_jokes_jsonl(input_jsonl)
    print(f"Loaded jokes: {len(jokes)}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    global_seen: set[Tuple[str, str]] = set()  # (target, unordered_pair_key)

    written = 0
    skipped = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, (setup, punchline) in enumerate(jokes):
            target = make_one_line_target(setup, punchline)

            if not target:
                skipped += 1
                continue

            if max_words is not None and count_words(target) > max_words:
                skipped += 1
                continue

            # optional topic line from setup: first 12 words
            topic_line = ""
            if include_topic_line:
                topic_line = " ".join(setup.split()[:12])

            per_joke_seen_pairs: set[str] = set()

            # try to produce up to samples_per_joke unique pairs
            for k in range(samples_per_joke):
                rnd = random.Random(seed + i * 10_000 + k)

                chosen_pair: Optional[Tuple[str, str]] = None

                for _attempt in range(max_pair_attempts_per_sample):
                    anchors = choose_two_anchors_with_mix(
                        setup,
                        punchline,
                        rnd=rnd,
                        prefer_common_nouns=prefer_common_nouns,
                    )
                    if anchors is None:
                        continue

                    w1_raw, w2_raw = anchors
                    w1 = pb.safe_word(w1_raw)
                    w2 = pb.safe_word(w2_raw)
                    if not w1 or not w2:
                        continue
                    if w1 == w2:
                        continue

                    # ensure anchors actually appear in the one-line target exactly (case-sensitive)
                    if not (contains_exact(target, w1) and contains_exact(target, w2)):
                        continue

                    unordered_key = "||".join(sorted([w1, w2], key=lambda x: x.lower()))

                    # per-joke duplicate avoidance
                    if unordered_key in per_joke_seen_pairs:
                        continue

                    # global duplicate avoidance (same target with same unordered pair)
                    global_key = (target, unordered_key)
                    if global_key in global_seen:
                        continue

                    chosen_pair = (w1, w2)
                    per_joke_seen_pairs.add(unordered_key)
                    global_seen.add(global_key)
                    break

                if chosen_pair is None:
                    # cannot find a new pair for this sample
                    skipped += 1
                    continue

                word1, word2 = chosen_pair

                messages = build_final_messages(
                    word1=word1,
                    word2=word2,
                    plan_text=DEFAULT_PLAN,
                    topic_line=topic_line,
                    include_noun_cards=include_noun_cards,
                )

                row = {
                    "id": f"{i}_{k}",
                    "word1": word1,
                    "word2": word2,
                    "messages": messages,  # trainer applies chat template at training time
                    "target": target,      # one-line: setup + space + punchline
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote rows: {written}")
    print(f"Skipped rows: {skipped}")
    print(f"Output: {output_jsonl}")


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
build_dataset_1_word_inclusion_messages_only(
    input_jsonl=Path("../data/merged_sft_jokes.jsonl"),
    output_jsonl=Path("../data/dataset1_word_inclusion_messages_only.jsonl"),
    samples_per_joke=2,
    seed=1337,
    include_noun_cards=True,   # set False if you want to avoid external facts fetching
    include_topic_line=True,
    prefer_common_nouns=True,
    max_words=30,
    max_pair_attempts_per_sample=120,
)
