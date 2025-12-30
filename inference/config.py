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

    vl_b1_scenes_csv: Path = data_dir / "task-b1.scenes.csv"
    vl_b2_scenes_csv: Path = data_dir / "task-b2.scenes.csv"


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

    # If True, noun chunks are considered as candidates, but they will be reduced
    # to "clean noun phrases" (compounds + head), and determiners/possessives removed.
    allow_noun_chunks: bool = True
    max_chunk_tokens: int = 3

    # Pair picking
    prefer_one_from_each: bool = True
    max_tries_per_pair: int = 40

    # Candidate text cleaning
    whitespace_pattern: str = r"\s+"
    normalize_curly_apostrophe: bool = True

    # IMPORTANT: This pattern is applied, but disallowed characters are replaced with SPACE
    # in the extractor (to prevent concatenation like familyMom).
    phrase_allowed_chars_pattern: str = r"[^A-Za-z0-9 \-']"

    # --- New: stricter "clean noun" controls ---
    strip_leading_determiners: bool = True              # the/a/an/this/that/these/those
    strip_leading_possessives: bool = True              # my/your/his/her/our/their/its
    strip_leading_slang_possessives: bool = True        # me/ma (joke corpora often uses these)
    split_camelcase: bool = True                        # familyMom -> family Mom

    # Hyphens and digits
    split_hyphens_to_space: bool = True                 # x-ray -> x ray, -year-old -> year old
    reject_if_contains_digit: bool = True               # 3 wise men, 50 shades, etc.
    reject_if_starts_with_punct: bool = True            # -You'll

    # Enforce tokens are "word-ish"
    require_alpha_tokens_only: bool = True              # after stripping apostrophes: only letters in each token
    reject_contraction_like: bool = True                # You'll, I'm, we're, etc.

    # Normalization preference for Wikipedia: use lemma for common nouns
    use_lemma_for_common_nouns: bool = True

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

    # Weaker matching options
    allow_singular_from_plural: bool = True   # "eggs" anchor will also accept "egg"
    allow_verb_ed: bool = True                # "spray" anchor will also accept "sprayed"
    allow_verb_ing: bool = True               # "spray" anchor will also accept "spraying"
    


# =============================================================================
# Prompt text (unchanged; kept here for single source of truth)
# =============================================================================

@dataclass(frozen=True)
class PromptTexts:
    system_plan: str = (
        "You are a humor generation assistant. Follow instructions exactly. Output must be machine-parseable. One line. No extra text, no markdown."
    )

    system_final: str = (
        "You are a humor generation assistant. Follow instructions exactly. Output only the final required answer. One line. No extra text, no markdown."
    )
    two_words_pun_plan_task: str = (
        "### Pun and wordplay\n"
        "Definition: lexical pivot supports two readings; punchline forces the hidden one. Not irony or satire: language mechanics.\n"
        "Inputs: mode=two_words; nouns=[noun1,noun2] exact; contexts in FACTS.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences).\n"
        "Schema:\n"
        '{"humor_type":"pun","mode":"two_words","nouns":["<noun1>","<noun2>"],'
        '"noun_terms":{"noun1_terms":[...],"noun2_terms":[...]},'
        '"shared_terms":[...],'
        '"pivot_candidates":[{"pivot":"...","dual_readings":["surface","hidden"],"link":"..."}],'
        '"pun_core":{"combination_proposition":"...","inferred_setup_candidates":[...],"hidden_setup_candidates":[...]},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: noun_terms 5-10 each; shared_terms 0-4; pivot_candidates 5-8; inferred_setup 2-5; hidden_setup 2-4.\n"
    )

    caption_mm_b1_pun_plan_task: str = (
        "### Pun and wordplay\n"
        "Definition: lexical pivot supports two readings; punchline forces the hidden one. Not irony or satire: language mechanics.\n"
        "Inputs: mode=image_caption_b1; image_facts present; no prompt_text; pick nouns as two concrete visible nouns.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences).\n"
        "Schema:\n"
        '{"humor_type":"pun","mode":"image_caption_b1","max_words":20,"nouns":["<noun1>","<noun2>"],'
        '"noun_terms":{"noun1_terms":[...],"noun2_terms":[...]},'
        '"shared_terms":[...],'
        '"pivot_candidates":[{"pivot":"...","dual_readings":["surface","hidden"],"link":"..."}],'
        '"pun_core":{"headline_knowledge":[...],"combination_proposition":"...","inferred_setup_candidates":[...],"hidden_setup_candidates":[...]},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: noun_terms 5-8 each; shared_terms 0-4; pivot_candidates 5-7; headline_knowledge 0-2; inferred_setup 2-4; hidden_setup 2-4. Must support final caption <= 20 words.\n"
    )

    caption_mm_b2_pun_plan_task: str = (
        "### Pun and wordplay\n"
        "Definition: lexical pivot supports two readings; punchline forces the hidden one. Not irony or satire: language mechanics.\n"
        "Inputs: mode=image_caption_b2; image_facts present; prompt_text provided; complete prompt_text with humorous content; pick nouns as two concrete visible nouns.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences).\n"
        "Schema:\n"
        '{"humor_type":"pun","mode":"image_caption_b2","max_words":20,"prompt_text":"<prompt_text>","nouns":["<noun1>","<noun2>"],'
        '"noun_terms":{"noun1_terms":[...],"noun2_terms":[...]},'
        '"shared_terms":[...],'
        '"pivot_candidates":[{"pivot":"...","dual_readings":["surface","hidden"],"link":"..."}],'
        '"pun_core":{"headline_knowledge":[...],"combination_proposition":"...","inferred_setup_candidates":[...],"hidden_setup_candidates":[...]},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: noun_terms 5-8 each; shared_terms 0-4; pivot_candidates 5-7; headline_knowledge 0-2; inferred_setup 2-4; hidden_setup 2-4. Must support final completion <= 20 words total.\n"
    )

    headline_satire_plan_task: str = (
        "### Satire\n"
        "Definition: mock news framing that critiques public life (politics, media, corporate/platform, technology, culture) via exaggerated reporting.\n"
        "Not pun or irony: core is public critique in news voice, not lexical ambiguity or everyday speaker reversal.\n"
        "Inputs: mode=headline; headline provided; nouns=[noun1,noun2] available; contexts in FACTS.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences), except narrative_knowledge.\n"
        "Schema:\n"
        '{"humor_type":"satire","mode":"headline","headline":"<headline>","nouns":["<noun1>","<noun2>"],'
        '"derived_framing":{"domain":"politics|media|corporate/platform|technology|culture",'
        '"target_candidates":[...],"move_candidates":[...],"critique_candidates":[...]},'
        '"narrative_knowledge":[...],'
        '"satire_core":{"combination_proposition":"..."},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: target_candidates 2-3; move_candidates 2-3; critique_candidates 2-3; narrative_knowledge 2-4 (1 sentence each); combination_proposition 0-1 (1 sentence).\n"
    )

    two_words_irony_plan_task: str = (
        "### Irony\n"
        "Definition: intent mismatch (said ≠ meant), e.g., praise for something clearly bad or calm understatement for something annoying.\n"
        "Not pun or satire: no wordplay pivot needed; no public target or mock-news voice required.\n"
        "Inputs: mode=two_words; nouns=[noun1,noun2] exact; contexts in FACTS.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences), except everyday_knowledge.\n"
        "Schema:\n"
        '{"humor_type":"irony","mode":"two_words","nouns":["<noun1>","<noun2>"],'
        '"derived_situation":{"domain":"daily life|work/school|services|home|health/energy",'
        '"situation_candidates":[...],"negative_reality_candidates":[...],"positive_utterance_candidates":[...]},'
        '"everyday_knowledge":[...],'
        '"irony_core":{"combination_proposition":"..."},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: situation_candidates 2-3; negative_reality_candidates 2-3; positive_utterance_candidates 2-3; everyday_knowledge 2-4 (1 sentence each); combination_proposition 0-1 (1 sentence).\n"
    )

    headline_irony_plan_task: str = (
        "### Irony\n"
        "Definition: intent mismatch (said ≠ meant), e.g., praise for something clearly bad or calm understatement for something annoying.\n"
        "Not pun or satire: no wordplay pivot needed; no public target or mock-news voice required.\n"
        "Inputs: mode=headline; headline provided; nouns=[noun1,noun2] available; contexts in FACTS.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences), except everyday_knowledge.\n"
        "Schema:\n"
        '{"humor_type":"irony","mode":"headline","headline":"<headline>","nouns":["<noun1>","<noun2>"],'
        '"derived_situation":{"domain":"daily life|work/school|services|home|health/energy",'
        '"situation_candidates":[...],"negative_reality_candidates":[...],"positive_utterance_candidates":[...]},'
        '"everyday_knowledge":[...],'
        '"irony_core":{"combination_proposition":"..."},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: situation_candidates 2-3; negative_reality_candidates 2-3; positive_utterance_candidates 2-3; everyday_knowledge 2-4 (1 sentence each); combination_proposition 0-1 (1 sentence).\n"
    )

    caption_mm_b1_irony_plan_task: str = (
        "### Irony\n"
        "Definition: intent mismatch (said ≠ meant), e.g., praise for something clearly bad or calm understatement for something annoying.\n"
        "Not pun or satire: no wordplay pivot needed; no public target or mock-news voice required.\n"
        "Inputs: mode=image_caption_b1; image_facts present; no prompt_text; pick nouns as two concrete visible nouns.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences), except everyday_knowledge.\n"
        "Schema:\n"
        '{"humor_type":"irony","mode":"image_caption_b1","max_words":20,"nouns":["<noun1>","<noun2>"],'
        '"derived_situation":{"domain":"daily life|work/school|services|home|health/energy",'
        '"situation_candidates":[...],"negative_reality_candidates":[...],"positive_utterance_candidates":[...]},'
        '"everyday_knowledge":[...],'
        '"irony_core":{"combination_proposition":"..."},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: situation_candidates 2-3; negative_reality_candidates 2-3; positive_utterance_candidates 2-3; everyday_knowledge 2-4 (1 sentence each); combination_proposition 0-1 (1 sentence). Must support final caption <= 20 words.\n"
    )

    caption_mm_b2_irony_plan_task: str = (
        "### Irony\n"
        "Definition: intent mismatch (said ≠ meant), e.g., praise for something clearly bad or calm understatement for something annoying.\n"
        "Not pun or satire: no wordplay pivot needed; no public target or mock-news voice required.\n"
        "Inputs: mode=image_caption_b2; image_facts present; prompt_text provided; complete prompt_text with humorous content; pick nouns as two concrete visible nouns.\n"
        "Return ONE LINE JavaScript Object Notation only, no extra keys. Keep strings short (phrases, not sentences), except everyday_knowledge.\n"
        "Schema:\n"
        '{"humor_type":"irony","mode":"image_caption_b2","max_words":20,"prompt_text":"<prompt_text>","nouns":["<noun1>","<noun2>"],'
        '"derived_situation":{"domain":"daily life|work/school|services|home|health/energy",'
        '"situation_candidates":[...],"negative_reality_candidates":[...],"positive_utterance_candidates":[...]},'
        '"everyday_knowledge":[...],'
        '"irony_core":{"combination_proposition":"..."},'
        '"output_blueprint":{"format":"one-liner|question_answer"}}\n'
        "Sizes: situation_candidates 2-3; negative_reality_candidates 2-3; positive_utterance_candidates 2-3; everyday_knowledge 2-4 (1 sentence each); combination_proposition 0-1 (1 sentence). Must support final completion <= 20 words total.\n"
    )

    two_words_final_task: str = (
        "Write one short joke in English. One line only.\n"
        "Must include BOTH words exactly: <noun1> and <noun2>.\n"
        "Follow PLAN strictly (humor_type + candidates). No explanation.\n"
        "PLAN: {plan}\n"
    )

    headline_final_task: str = (
        "Write one short joke in English. One line only.\n"
        "Must be clearly related to the headline (do not copy it verbatim).\n"
        "Follow PLAN strictly (humor_type + candidates). No explanation.\n"
        "HEADLINE: {headline}\n"
        "PLAN: {plan}\n"
    )

    caption_mm_b1_final_task: str = (
        "Write one humorous caption in English for the GIF. One line only. Max 20 words.\n"
        "Follow PLAN strictly (humor_type + candidates). No explanation.\n"
        "IMAGE_FACTS: {image_facts}\n"
        "PLAN: {plan}\n"
    )

    caption_mm_b2_final_task: str = (
        "Complete PROMPT_TEXT in English using the GIF. Output must start with PROMPT_TEXT exactly.\n"
        "One line only. Max 20 words total (including PROMPT_TEXT).\n"
        "Follow PLAN strictly (humor_type + candidates). No explanation.\n"
        "PROMPT_TEXT: {prompt_text}\n"
        "IMAGE_FACTS: {image_facts}\n"
        "PLAN: {plan}\n"
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

@dataclass(frozen=True)
class VLSceneExtractorSettings:
    # Model
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    attn_implementation: str = "sdpa"

    # Decoding / retries
    max_tries: int = 6
    max_new_tokens: int = 160
    temperature: float = 0.3
    top_p: float = 0.9

    # Output column names (CSV)
    scene_col: str = "scene"  # keep "scene" for now (runner compatibility)
    noun1_col: str = "noun1"
    noun2_col: str = "noun2"

    # New: JSON keys aligned with plan prompts
    json_facts_key: str = "image_facts"
    json_nouns_key: str = "nouns"

    system_prompt: str = (
        "Return ONE LINE JSON only. No markdown. No extra keys.\n"
        "Schema:\n"
        '{"image_facts":"...","nouns":["...","..."]}\n'
        "Rules:\n"
        "- image_facts: neutral, concrete, visible-only; 1 sentence; 8-25 words; no jokes.\n"
        "- nouns: exactly 2 distinct visible common nouns; lowercase a-z; single word; avoid generic (person, people, thing, object, stuff).\n"
    )

    user_prompt: str = "Analyze the images and produce the JSON."

    repair_user_template: str = (
        "Fix your output to match Schema and Rules. Return ONE LINE JSON only.\n"
        "Previous: {previous}\n"
    )

    banned_nouns_extra: Tuple[str, ...] = ("object", "thing", "stuff", "person", "people")


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

    vl_scene_extractor: VLSceneExtractorSettings = field(default_factory=VLSceneExtractorSettings)

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

@dataclass(frozen=True)
class ResponseEvaluatorConfig:
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    required_words: RequiredWordsSettings = field(default_factory=RequiredWordsSettings)