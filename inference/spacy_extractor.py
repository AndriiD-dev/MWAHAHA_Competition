from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from config import RequiredWordsSettings, SpacySettings


@dataclass
class SpacyAnchorExtractor:
    """spaCy anchor extraction with all tunables injected via SpacySettings."""

    settings: SpacySettings
    generic_nouns: Set[str] = field(default_factory=set)
    extra_stopwords: Set[str] = field(default_factory=set)

    _nlp: Optional[Language] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ws_re = re.compile(self.settings.whitespace_pattern)

    def load(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.load(self.settings.model_name, disable=list(self.settings.disable_components))
        return self._nlp

    def normalize_one_line(self, text: str) -> str:
        text = "" if text is None else str(text)
        return self._ws_re.sub(" ", text).strip()

    def safe_phrase_preserve_case(self, text: str) -> str:
        t = self.normalize_one_line(text)
        if not t:
            return ""
        if self.settings.normalize_curly_apostrophe:
            t = t.replace("’", "'")
        t = re.sub(self.settings.phrase_allowed_chars_pattern, "", t)
        t = self._ws_re.sub(" ", t).strip()
        return t

    def parse(self, text: str) -> Doc:
        nlp = self.load()
        t = self.normalize_one_line(text)
        if not t:
            return nlp.make_doc("")
        if len(t) > self.settings.max_text_chars:
            t = t[: self.settings.max_text_chars]
        return nlp(t)

    def _is_bad_candidate(self, text: str) -> bool:
        if not text:
            return True
        low = text.lower()
        if low in self.generic_nouns:
            return True
        if low in self.extra_stopwords:
            return True
        return False

    def extract_candidates(self, text: str) -> List[str]:
        doc = self.parse(text)
        out: List[str] = []
        seen = set()

        for tok in doc:
            if tok.pos_ not in self.settings.allowed_parts_of_speech:
                continue
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            if len(tok.text) < self.settings.min_token_chars:
                continue
            if tok.is_stop:
                continue

            cand = self.safe_phrase_preserve_case(tok.text)
            if not cand or self._is_bad_candidate(cand):
                continue

            key = cand.lower()
            if key not in seen:
                seen.add(key)
                out.append(cand)

        if self.settings.allow_noun_chunks:
            for chunk in doc.noun_chunks:
                toks = [t for t in chunk if not (t.is_space or t.is_punct)]
                if not toks:
                    continue
                if len(toks) > self.settings.max_chunk_tokens:
                    continue
                if chunk.root.pos_ not in self.settings.allowed_parts_of_speech:
                    continue

                text_chunk = self.safe_phrase_preserve_case(chunk.text)
                if not text_chunk or self._is_bad_candidate(text_chunk):
                    continue

                key = text_chunk.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(text_chunk)

        return out

    def base_form_for_similarity(self, phrase: str) -> str:
        p = self.safe_phrase_preserve_case(phrase)
        if not p:
            return ""

        if self.settings.baseform_strip_possessive:
            p = re.sub(r"(?:'s)$", "", p, flags=re.IGNORECASE)

        parts = p.split()
        last = parts[-1]
        low = last.lower()

        minlen = int(self.settings.baseform_min_len_for_plural_rules)

        if self.settings.baseform_variant_ies_to_y and low.endswith("ies") and len(low) > minlen:
            low = low[:-3] + "y"
        elif self.settings.baseform_variant_strip_plural_es and low.endswith("es") and len(low) > minlen:
            low = low[:-2]
        elif self.settings.baseform_variant_strip_plural_s and low.endswith("s") and len(low) > minlen and not low.endswith("ss"):
            low = low[:-1]

        parts[-1] = low
        return " ".join(parts).lower()

    def pick_two_anchors(self, setup: str, punchline: str, rng: Optional[random.Random] = None) -> Optional[Tuple[str, str]]:
        rng = rng or random.Random()

        setup_c = self.extract_candidates(setup)
        punch_c = self.extract_candidates(punchline)

        if not setup_c and not punch_c:
            return None

        def pick_one(pool: Sequence[str]) -> Optional[str]:
            return rng.choice(pool) if pool else None

        for _ in range(self.settings.max_tries_per_pair):
            if self.settings.prefer_one_from_each:
                a = pick_one(setup_c) or pick_one(punch_c)
                b = pick_one(punch_c) or pick_one(setup_c)
            else:
                combined = setup_c + punch_c
                if len(combined) < 2:
                    return None
                a, b = rng.sample(combined, 2)

            if not a or not b:
                continue
            if a.lower() == b.lower():
                continue
            if self.base_form_for_similarity(a) == self.base_form_for_similarity(b):
                continue
            return (a, b)

        return None


@dataclass
class RequiredWordsChecker:
    """Required words checker with all tunables injected via RequiredWordsSettings."""

    settings: RequiredWordsSettings

    def __post_init__(self) -> None:
        self._ws_re = re.compile(self.settings.whitespace_pattern)

    def normalize_one_line(self, text: str) -> str:
        text = "" if text is None else str(text)
        return self._ws_re.sub(" ", text).strip()

    def safe_phrase_preserve_case(self, text: str) -> str:
        t = self.normalize_one_line(text)
        if not t:
            return ""
        if self.settings.normalize_curly_apostrophe:
            t = t.replace("’", "'")
        t = re.sub(self.settings.phrase_allowed_chars_pattern, "", t)
        t = self._ws_re.sub(" ", t).strip()
        return t

    def _token_plural_possessive_pattern(self, token: str) -> str:
        base = token.lower()
        if not base:
            return r"a^"

        if len(base) <= int(self.settings.strict_if_len_leq) or not base.isalpha():
            return re.escape(base)

        core = re.escape(base)

        if base.endswith("y") and len(base) > 3 and self.settings.allow_plural_ies:
            stem = re.escape(base[:-1])
            plural = rf"(?:y|ies)"
            core = rf"{stem}{plural}"
        else:
            plural_parts = []
            if self.settings.allow_plural_s:
                plural_parts.append("s")
            if self.settings.allow_plural_es:
                plural_parts.append("es")
            if plural_parts:
                core = core + rf"(?:{'|'.join(plural_parts)})?"
            else:
                core = core

        if self.settings.allow_possessive:
            core += r"(?:'s|’s|s'|s’)?"

        return core

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
        full = rf"{self.settings.boundary_left}{head_pat}{last_pat}{self.settings.boundary_right}"
        return re.compile(full, flags=re.IGNORECASE)

    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        t = self.normalize_one_line(text)
        if not t:
            return False

        w1 = self.safe_phrase_preserve_case(word1)
        w2 = self.safe_phrase_preserve_case(word2)
        if not w1 or not w2:
            return False

        return (self.boundary_pattern(w1).search(t) is not None) and (self.boundary_pattern(w2).search(t) is not None)
