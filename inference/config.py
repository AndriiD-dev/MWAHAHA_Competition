from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Sequence, Tuple


# =============================================================================
# Paths
# =============================================================================

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    wiki_cache_path: Path = data_dir / "wiki_cache.json"


# =============================================================================
# Wikipedia reader settings (ALL tunables live here)
# =============================================================================

@dataclass(frozen=True)
class WikiSettings:
    # Endpoints
    summary_url_template: str = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
    search_url: str = "https://en.wikipedia.org/w/api.php"

    # Networking
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "MWAHAHA/1.0 (contact: dardemtum@gmail.com) humor-generation"
    })
    timeout_seconds: float = 12.0
    sleep_seconds: float = 0.10
    search_limit: int = 6

    # Cache
    cache_max_entries: int = 50_000

    # Text normalization and title sanitization
    whitespace_pattern: str = r"\s+"
    title_allowed_chars_pattern: str = r"[^A-Za-z0-9 \-']"  # characters to REMOVE
    normalize_curly_apostrophe: bool = True

    # Disambiguation detection / wipeout
    disambiguation_page_type: str = "disambiguation"
    disambiguation_phrases: Tuple[str, ...] = ("may refer to", "commonly refers to")

    # Variants generation
    variant_add_lower: bool = True
    variant_add_title: bool = True
    variant_strip_plural_s: bool = True
    variant_strip_plural_es: bool = True
    variant_ies_to_y: bool = True
    variant_min_len_for_plural_rules: int = 4

    # FACTS / microcards formatting
    facts_extract_cap_chars: int = 260
    microcard_first_sentence_cap_chars: int = 160
    microcard_keywords_max: int = 6


# =============================================================================
# spaCy extraction settings (ALL tunables live here)
# =============================================================================

@dataclass(frozen=True)
class SpacySettings:
    # Model / pipeline
    model_name: str = "en_core_web_sm"
    disable_components: Tuple[str, ...] = ("ner",)
    max_text_chars: int = 50_000

    # Candidate selection
    allowed_parts_of_speech: Tuple[str, ...] = ("NOUN", "PROPN")
    min_token_chars: int = 2
    allow_noun_chunks: bool = True
    max_chunk_tokens: int = 3

    # Pair picking
    prefer_one_from_each: bool = True
    max_tries_per_pair: int = 40

    # Candidate text cleaning
    whitespace_pattern: str = r"\s+"
    normalize_curly_apostrophe: bool = True
    phrase_allowed_chars_pattern: str = r"[^A-Za-z0-9 \-']"  # characters to REMOVE

    # Similarity guard (base form heuristics)
    baseform_strip_possessive: bool = True
    baseform_variant_strip_plural_s: bool = True
    baseform_variant_strip_plural_es: bool = True
    baseform_variant_ies_to_y: bool = True
    baseform_min_len_for_plural_rules: int = 4


# =============================================================================
# Required words checker settings (moved out of spacy_extractor.py)
# =============================================================================

@dataclass(frozen=True)
class RequiredWordsSettings:
    # Boundaries
    boundary_left: str = r"(?<![A-Za-z0-9])"
    boundary_right: str = r"(?![A-Za-z0-9])"

    # Cleaning
    whitespace_pattern: str = r"\s+"
    normalize_curly_apostrophe: bool = True
    phrase_allowed_chars_pattern: str = r"[^A-Za-z0-9 \-']"  # characters to REMOVE

    # Matching behavior
    strict_if_len_leq: int = 3

    allow_plural_s: bool = True
    allow_plural_es: bool = True
    allow_plural_ies: bool = True
    allow_possessive: bool = True  # 's, ’s, s', s’


# =============================================================================
# Prompt text (unchanged; kept here for single source of truth)
# =============================================================================

@dataclass(frozen=True)
class PromptTexts:
    final_common: str = (
        "You are a stand-up comedian. Write ONE original joke in English.\n"
        "Return exactly one line under 30 words. No preface, no explanation, no emojis.\n"
        "Avoid slurs or hate toward protected groups. Avoid explicit sexual content and graphic violence. "
        "Mild innuendo, flirting, and cartoonish (non-graphic) mishaps are allowed.\n"
        "Do not apologize or refuse; if the topic is sensitive, pivot to wordplay, absurdity, "
        "or self-deprecation and still deliver a joke.\n"
        "You may receive FACTS and PLAN blocks; use them only to guide the joke. Do not quote them."
    )

    plan_common: str = (
        "You are a stand-up comedian. First write a short private plan for a joke.\n"
        "Do not write the joke yet. Output only the plan.\n"
        "Keep it concise and practical. No preface, no emojis.\n"
        "You may receive a FACTS block; use it only to understand the anchor words and do not quote it."
    )

    two_words_plan_task: str = (
        "Task: Create a short plan for a one-line joke that uses BOTH required words naturally.\n"
        "Required words: '{word1}' and '{word2}'.\n"
        "Output format: a single-line JSON object with keys "
        "\"scenario\", \"misdirection\", \"word_placement\", \"device\".\n"
        "Do not include the final joke."
    )

    two_words_final_task: str = (
        "Write the final joke now.\n"
        "Constraints: must include '{word1}' and '{word2}' exactly as written.\n"
        "Output ONLY the joke."
    )

    title_plan_task: str = (
        "Headline: {headline}\n"
        "Anchor nouns: '{noun1}', '{noun2}'.\n\n"
        "Task: Create a short plan for a one-line joke inspired by the headline.\n"
        "The final joke must include BOTH anchor nouns exactly as written.\n"
        "Output format: a single-line JSON object with keys "
        "\"angle\", \"misdirection\", \"word_placement\", \"device\".\n"
        "Do not include the final joke."
    )

    title_final_task: str = (
        "Headline: {headline}\n"
        "Anchor nouns (must appear): '{noun1}', '{noun2}'.\n"
        "PLAN (do not quote): {plan}\n\n"
        "Write the final joke now.\n"
        "Constraints: one line, under 30 words, include BOTH anchor nouns (you may pluralize or use possessive).\n"
        "Output ONLY the joke."
    )


# =============================================================================
# Microcards (used by PromptBuilder)
# =============================================================================

@dataclass(frozen=True)
class MicrocardSettings:
    domain_hints: Dict[str, Sequence[str]] = field(default_factory=lambda: {
        "biology": ("species", "genus", "family", "organism", "plant", "animal", "fungus", "bacteria"),
        "technology": ("software", "hardware", "computer", "system", "device", "network", "protocol", "algorithm"),
        "science": ("physics", "chemistry", "mathematics", "astronomy", "geology", "theory", "experiment"),
        "geography": ("city", "country", "region", "river", "mountain", "island", "capital", "province"),
        "arts": ("film", "music", "band", "album", "novel", "painting", "artist", "composer"),
        "sports": ("team", "league", "season", "tournament", "player", "coach", "championship"),
    })

    keyword_stop: FrozenSet[str] = frozenset({
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while",
        "to", "of", "in", "on", "for", "at", "by", "with", "from", "as",
        "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "this", "that", "these", "those",
    })


# =============================================================================
# Top-level config object
# =============================================================================

@dataclass(frozen=True)
class PromptBuilderConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    wiki: WikiSettings = field(default_factory=WikiSettings)
    spacy: SpacySettings = field(default_factory=SpacySettings)
    required_words: RequiredWordsSettings = field(default_factory=RequiredWordsSettings)
    prompts: PromptTexts = field(default_factory=PromptTexts)
    microcards: MicrocardSettings = field(default_factory=MicrocardSettings)

    generic_nouns: FrozenSet[str] = frozenset({
        "thing", "things", "stuff", "someone", "anyone", "everyone",
        "person", "people", "man", "men", "woman", "women",
        "time", "day", "week", "month", "year",
        "place", "home", "house", "room",
        "job", "work", "boss", "company",
        "way", "kind", "sort", "part", "case", "point", "problem", "idea",
        "joke", "story",
    })

    extra_stopwords: FrozenSet[str] = frozenset({
        "yeah", "okay", "ok", "lol",
    })
