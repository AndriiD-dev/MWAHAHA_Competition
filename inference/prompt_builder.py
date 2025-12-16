from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from config import DEFAULT_CONFIG, PromptBuilderConfig

from spacy_extractor import SpacyAnchorExtractor
from wiki_reader import WikipediaReader


_WS_RE = re.compile(r"\s+")


def normalize_one_line(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def safe_word(word: str) -> str:
    """Trim whitespace only. Do not destroy internal casing."""
    return normalize_one_line(word).strip()


@dataclass
class RequiredWordsChecker:
    """Robust word/phrase presence checker.

    Properties:
    - case-insensitive match in the *generated text*
    - preserves internal casing in the *required words* (no lowercasing/sanitizing)
    - allows plural and possessive on the last token of a phrase
    - treats word boundaries as non-alphanumeric (so punctuation works)
    """

    boundary_left: str = r"(?<![A-Za-z0-9])"
    boundary_right: str = r"(?![A-Za-z0-9])"
    strict_if_len_leq: int = 3

    @staticmethod
    def safe_phrase_preserve_case(text: str) -> str:
        t = normalize_one_line(text)
        if not t:
            return ""
        # normalize curly apostrophe to straight for stable patterns
        t = t.replace("’", "'")
        # allow letters, digits, spaces, hyphens, apostrophes
        t = re.sub(r"[^A-Za-z0-9 \-']", "", t)
        t = _WS_RE.sub(" ", t).strip()
        return t

    def _token_plural_possessive_pattern(self, token: str) -> str:
        base = token.lower()
        if not base:
            return r"a^"

        # strict for short tokens / acronyms / mixed tokens
        if len(base) <= self.strict_if_len_leq or not base.isalpha():
            return re.escape(base)

        if base.endswith("y") and len(base) > 3:
            stem = re.escape(base[:-1])
            core = rf"{stem}(?:y|ies)"
        else:
            core = re.escape(base) + r"(?:s|es)?"

        # possessive (straight or curly); allow plural possessive too
        core += r"(?:'s|’s|s'|s’)?"
        return core

    @lru_cache(maxsize=4096)
    def boundary_pattern(self, phrase: str) -> re.Pattern:
        p = self.safe_phrase_preserve_case(phrase)
        if not p:
            return re.compile(r"a^")

        parts = p.split()
        if not parts:
            return re.compile(r"a^")

        head_tokens = parts[:-1]
        last_token = parts[-1]

        head_pat = ""
        if head_tokens:
            head_pat = r"\s+".join(re.escape(x.lower()) for x in head_tokens) + r"\s+"

        last_pat = self._token_plural_possessive_pattern(last_token)

        full = rf"{self.boundary_left}{head_pat}{last_pat}{self.boundary_right}"
        return re.compile(full, flags=re.IGNORECASE)

    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        t = normalize_one_line(text)
        if not t:
            return False

        w1 = self.safe_phrase_preserve_case(word1)
        w2 = self.safe_phrase_preserve_case(word2)
        if not w1 or not w2:
            return False

        return (self.boundary_pattern(w1).search(t) is not None) and (
            self.boundary_pattern(w2).search(t) is not None
        )


@dataclass
class PromptBuilder:
    """Single source of truth for prompt construction.

    - uses WikipediaReader for microcard FACTS
    - uses SpacyAnchorExtractor for headline noun selection
    - centralizes normalization and required-word validation
    """

    config: PromptBuilderConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    wiki: WikipediaReader | None = None
    spacy: SpacyAnchorExtractor | None = None
    rng: random.Random = field(default_factory=random.Random)

    checker: RequiredWordsChecker = field(default_factory=RequiredWordsChecker)

    def __post_init__(self) -> None:
        if self.wiki is None:
            self.wiki = WikipediaReader(
                summary_url_template=self.config.wiki.summary_url_template,
                search_url=self.config.wiki.search_url,
                headers=dict(self.config.wiki.headers),
                timeout_seconds=self.config.wiki.timeout_seconds,
                sleep_seconds=self.config.wiki.sleep_seconds,
                search_limit=self.config.wiki.search_limit,
                cache_max_entries=self.config.wiki.cache_max_entries,
            )
            try:
                self.wiki.load_cache(self.config.paths.wiki_cache_path)
            except Exception:
                pass

        if self.spacy is None:
            self.spacy = SpacyAnchorExtractor(
                model_name=self.config.spacy.model_name,
                disable_components=self.config.spacy.disable_components,
                max_text_chars=self.config.spacy.max_text_chars,
                allowed_parts_of_speech=self.config.spacy.allowed_parts_of_speech,
                min_token_chars=self.config.spacy.min_token_chars,
                allow_noun_chunks=self.config.spacy.allow_noun_chunks,
                max_chunk_tokens=self.config.spacy.max_chunk_tokens,
                generic_nouns=set(x.lower() for x in self.config.generic_nouns),
                extra_stopwords=set(x.lower() for x in self.config.extra_stopwords),
                prefer_one_from_each=self.config.spacy.prefer_one_from_each,
                max_tries_per_pair=self.config.spacy.max_tries_per_pair,
            )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        return self.checker.required_words_present(text, word1, word2)

    # ------------------------------------------------------------------
    # Wiki -> microcards
    # ------------------------------------------------------------------
    def _guess_domain(self, extract: str) -> str:
        t = extract.lower()
        for domain, hints in self.config.microcards.domain_hints.items():
            if any(h in t for h in hints):
                return domain
        return ""

    def microcard_from_extract(self, word: str, extract: str) -> Dict[str, object]:
        w = safe_word(word)
        e = normalize_one_line(extract)

        if not e:
            return {"word": w, "what": "", "domain": "", "keywords": []}

        first = re.split(r"(?<=[.!?])\s+", e, maxsplit=1)[0]
        what = normalize_one_line(first)
        cap = self.config.wiki.microcard_first_sentence_cap_chars
        if len(what) > cap:
            what = what[: max(0, cap - 3)].rstrip() + "..."

        domain = self._guess_domain(e)

        tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", e)
        tokens_lower = [t.lower() for t in tokens]
        keywords: List[str] = []
        for t in tokens_lower:
            if (
                len(t) >= 5
                and t not in self.config.microcards.keyword_stop
                and t not in keywords
            ):
                keywords.append(t)
            if len(keywords) >= self.config.wiki.microcard_keywords_max:
                break

        return {"word": w, "what": what, "domain": domain, "keywords": keywords}

    def build_microcards_dict(self, word1: str, word2: str) -> Dict[str, Dict[str, object]]:
        w1 = safe_word(word1)
        w2 = safe_word(word2)

        assert self.wiki is not None
        e1 = self.wiki.get_wikipedia_extract(w1)
        e2 = self.wiki.get_wikipedia_extract(w2)

        c1 = self.microcard_from_extract(w1, e1)
        c2 = self.microcard_from_extract(w2, e2)

        return {
            w1: {
                "what": c1.get("what", ""),
                "domain": c1.get("domain", ""),
                "keywords": c1.get("keywords", []),
            },
            w2: {
                "what": c2.get("what", ""),
                "domain": c2.get("domain", ""),
                "keywords": c2.get("keywords", []),
            },
        }

    def format_facts_block(self, word1: str, word2: str) -> str:
        microcards = self.build_microcards_dict(word1, word2)
        w1 = safe_word(word1)
        w2 = safe_word(word2)

        def fmt_one(w: str) -> str:
            card = microcards.get(w, {})
            what = normalize_one_line(card.get("what", ""))
            domain = normalize_one_line(card.get("domain", ""))
            keywords = card.get("keywords", []) or []
            kw = ", ".join(list(keywords)[: self.config.wiki.microcard_keywords_max])

            parts: List[str] = []
            if what:
                parts.append(f"WHAT: {what}")
            if domain:
                parts.append(f"DOMAIN: {domain}")
            if kw:
                parts.append(f"KEYWORDS: {kw}")
            body = " | ".join(parts) if parts else ""
            return f"- {w}: {body}".rstrip()

        return "FACTS:\n" + fmt_one(w1) + "\n" + fmt_one(w2)

    def save_wiki_cache(self) -> None:
        assert self.wiki is not None
        try:
            self.wiki.save_cache(self.config.paths.wiki_cache_path)
        except Exception:
            # cache saving must never break inference
            pass

    # ------------------------------------------------------------------
    # spaCy: noun selection for headline task
    # ------------------------------------------------------------------
    def choose_two_nouns_from_headline(
        self,
        headline: str,
        *,
        seed: Optional[int] = None,
        prefer_distinct: bool = True,
    ) -> Tuple[str, str]:
        text = (headline or "").strip()
        if not text:
            return ("", "")

        rnd = random.Random(seed) if seed is not None else self.rng

        assert self.spacy is not None
        doc = self.spacy.parse(text)

        nouns: List[str] = []
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                lemma = (token.lemma_ or "").strip().lower()
                if lemma and lemma.isalpha() and len(lemma) > 2:
                    nouns.append(lemma)

        junk = {
            "thing",
            "stuff",
            "something",
            "anything",
            "everything",
            "someone",
            "anyone",
            "everyone",
        }
        nouns = [n for n in nouns if n not in junk]

        if nouns:
            unique = list(dict.fromkeys(nouns))
            if prefer_distinct and len(unique) >= 2:
                w1, w2 = rnd.sample(unique, 2)
                return (w1, w2)
            return (unique[0], unique[0])

        content: List[str] = []
        for token in doc:
            if token.pos_ in {"ADJ", "VERB"}:
                lemma = (token.lemma_ or "").strip().lower()
                if lemma and lemma.isalpha() and len(lemma) > 2:
                    content.append(lemma)

        if content:
            unique = list(dict.fromkeys(content))
            if len(unique) >= 2:
                return tuple(rnd.sample(unique, 2))  # type: ignore[return-value]
            return (unique[0], unique[0])

        tokens = [t.text.lower() for t in doc if t.text.isalpha() and len(t.text) > 2]
        if len(tokens) >= 2:
            return tuple(rnd.sample(tokens, 2))  # type: ignore[return-value]
        if len(tokens) == 1:
            return (tokens[0], tokens[0])
        return ("", "")

    # ------------------------------------------------------------------
    # Prompt building: Two words
    # ------------------------------------------------------------------
    def build_two_words_plan_messages(self, word1: str, word2: str) -> List[Dict[str, str]]:
        w1 = safe_word(word1)
        w2 = safe_word(word2)

        system = self.config.prompts.plan_common
        facts = self.format_facts_block(w1, w2)

        user = normalize_one_line(
            f"""{facts}

{self.config.prompts.two_words_plan_task.format(word1=w1, word2=w2)}"""
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_two_words_final_messages(self, word1: str, word2: str, plan_text: str) -> List[Dict[str, str]]:
        w1 = safe_word(word1)
        w2 = safe_word(word2)
        plan = normalize_one_line(plan_text)

        system = self.config.prompts.final_common
        facts = self.format_facts_block(w1, w2)

        user = normalize_one_line(
            f"""{facts}

PLAN (do not quote): {plan}

{self.config.prompts.two_words_final_task.format(word1=w1, word2=w2)}"""
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------------
    # Prompt building: Title (headline)
    # ------------------------------------------------------------------
    def build_title_plan_messages(self, headline: str, noun1: str, noun2: str) -> List[Dict[str, str]]:
        h = normalize_one_line(headline)
        n1 = safe_word(noun1)
        n2 = safe_word(noun2)

        system = self.config.prompts.plan_common
        facts = self.format_facts_block(n1, n2)

        user = normalize_one_line(
            f"""{facts}

{self.config.prompts.title_plan_task.format(headline=h, noun1=n1, noun2=n2)}"""
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_title_final_messages(
        self,
        headline: str,
        noun1: str,
        noun2: str,
        plan_text: str,
    ) -> List[Dict[str, str]]:
        h = normalize_one_line(headline)
        n1 = safe_word(noun1)
        n2 = safe_word(noun2)
        plan = normalize_one_line(plan_text)

        system = self.config.prompts.final_common
        facts = self.format_facts_block(n1, n2)

        user = normalize_one_line(
            f"""{facts}

{self.config.prompts.title_final_task.format(headline=h, noun1=n1, noun2=n2, plan=plan)}"""
        )

        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
