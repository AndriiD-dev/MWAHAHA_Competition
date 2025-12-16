from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Doc


# ---------------------------------------------------------------------------
# (1) spaCy anchor extraction + pair picking
# ---------------------------------------------------------------------------

@dataclass
class SpacyAnchorExtractor:
    """
    Extracts noun-like anchors from text using spaCy and picks 2 distinct anchors
    (with a similarity guard using a "base form").

    Everything that was previously hardcoded (model name, filters, stop lists,
    sampling settings) is now a class field.
    """

    # spaCy model / pipeline settings
    model_name: str = "en_core_web_sm"
    disable_components: Tuple[str, ...] = ("ner",)  # keep tagger+lemmatizer
    max_text_chars: int = 50_000

    # Candidate selection settings
    allowed_parts_of_speech: Tuple[str, ...] = ("NOUN", "PROPN")  # noun, proper noun
    min_token_chars: int = 2
    allow_noun_chunks: bool = True
    max_chunk_tokens: int = 3

    # Filtering lists (editable)
    generic_nouns: Set[str] = field(default_factory=set)
    extra_stopwords: Set[str] = field(default_factory=set)

    # Pair picking settings
    prefer_one_from_each: bool = True  # try 1 from setup and 1 from punchline
    max_tries_per_pair: int = 40

    # Internal cached model
    _nlp: Optional[Language] = field(default=None, init=False, repr=False)

    def load(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.load(self.model_name, disable=list(self.disable_components))
        return self._nlp

    @staticmethod
    def normalize_one_line(text: str) -> str:
        text = "" if text is None else str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _safe_phrase_preserve_case(text: str) -> str:
        """
        Keep internal casing exactly as written; remove only obviously-bad characters.
        Allows letters, digits, spaces, hyphens, apostrophes (straight+curly).
        """
        t = SpacyAnchorExtractor.normalize_one_line(text)
        if not t:
            return ""
        # normalize curly apostrophe to straight for storage consistency
        t = t.replace("’", "'")
        # drop everything else
        t = re.sub(r"[^A-Za-z0-9 \-']", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def parse(self, text: str) -> Doc:
        nlp = self.load()
        t = self.normalize_one_line(text)
        if not t:
            return nlp.make_doc("")
        if len(t) > self.max_text_chars:
            t = t[: self.max_text_chars]
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
        """
        Returns a de-duplicated list of anchor candidates (preserving original casing).
        """
        doc = self.parse(text)
        out: List[str] = []
        seen = set()

        # Token-level nouns / proper nouns
        for tok in doc:
            if tok.pos_ not in self.allowed_parts_of_speech:
                continue
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            if len(tok.text) < self.min_token_chars:
                continue
            # filter by spaCy stop words + your own stop list
            if tok.is_stop:
                continue

            cand = self._safe_phrase_preserve_case(tok.text)
            if not cand or self._is_bad_candidate(cand):
                continue

            key = cand.lower()
            if key not in seen:
                seen.add(key)
                out.append(cand)

        # Noun-chunk extraction (optional) for multi-token anchors
        if self.allow_noun_chunks:
            for chunk in doc.noun_chunks:
                # keep chunks reasonably short
                toks = [t for t in chunk if not (t.is_space or t.is_punct)]
                if not toks:
                    continue
                if len(toks) > self.max_chunk_tokens:
                    continue
                # ensure the head is noun-like
                if chunk.root.pos_ not in self.allowed_parts_of_speech:
                    continue

                text_chunk = self._safe_phrase_preserve_case(chunk.text)
                if not text_chunk or self._is_bad_candidate(text_chunk):
                    continue

                key = text_chunk.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(text_chunk)

        return out

    def base_form_for_similarity(self, phrase: str) -> str:
        """
        Used to reject pairs that are basically the same word.
        Example: "astronaut" vs "astronauts" -> same base form.
        """
        p = self._safe_phrase_preserve_case(phrase)
        if not p:
            return ""

        # strip possessive
        p = re.sub(r"(?:'s)$", "", p, flags=re.IGNORECASE)

        # if it is multi-token, compare only the last token’s lemma-like form
        parts = p.split()
        last = parts[-1]

        # heuristic singularization (fast, no extra dependencies)
        low = last.lower()
        if low.endswith("ies") and len(low) > 4:
            low = low[:-3] + "y"
        elif low.endswith("es") and len(low) > 4:
            low = low[:-2]
        elif low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
            low = low[:-1]

        parts[-1] = low
        return " ".join(parts).lower()

    def pick_two_anchors(
        self,
        setup: str,
        punchline: str,
        rng: Optional[random.Random] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Tries to pick two distinct anchors with different base forms.
        """
        rng = rng or random.Random()

        setup_c = self.extract_candidates(setup)
        punch_c = self.extract_candidates(punchline)

        if not setup_c and not punch_c:
            return None

        def pick_one(pool: Sequence[str]) -> Optional[str]:
            return rng.choice(pool) if pool else None

        for _ in range(self.max_tries_per_pair):
            if self.prefer_one_from_each:
                a = pick_one(setup_c) or pick_one(punch_c)
                b = pick_one(punch_c) or pick_one(setup_c)
            else:
                # allow both from anywhere
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


# ---------------------------------------------------------------------------
# (2) Improved "required words present" checker
# ---------------------------------------------------------------------------

@dataclass
class RequiredWordsChecker:
    """
    Upgraded version of your current regex-based required-words validation.

    Key differences versus the current code:
    - does NOT lowercase/sanitize away internal casing from the required words
      (so a required word like "iPhone" stays "iPhone" as a pattern source)
    - still matches case-insensitively in the generated text
    - allows plural + possessive variants
    - supports multi-token phrases (plural/possessive applied to the last token)
    """

    # What counts as a boundary (kept aligned with your existing approach)
    boundary_left: str = r"(?<![A-Za-z0-9])"
    boundary_right: str = r"(?![A-Za-z0-9])"

    # If a token is too short or contains non-letters, keep strict matching
    strict_if_len_leq: int = 3

    @staticmethod
    def normalize_one_line(text: str) -> str:
        text = "" if text is None else str(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def safe_phrase_preserve_case(text: str) -> str:
        t = RequiredWordsChecker.normalize_one_line(text)
        if not t:
            return ""
        # normalize curly apostrophe to straight for stable pattern building
        t = t.replace("’", "'")
        # allow letters, digits, spaces, hyphens, apostrophes
        t = re.sub(r"[^A-Za-z0-9 \-']", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _token_plural_possessive_pattern(self, token: str) -> str:
        """
        Build a pattern for a *single* final token allowing:
        - plural: s / es, y -> ies
        - possessive: 's, ’s, s', s’
        """
        base = token.lower()
        if not base:
            return r"a^"

        # strict for very short tokens / acronyms / mixed tokens
        if len(base) <= self.strict_if_len_leq or not base.isalpha():
            core = re.escape(base)
            return core

        if base.endswith("y") and len(base) > 3:
            stem = re.escape(base[:-1])
            core = rf"{stem}(?:y|ies)"
        else:
            core = re.escape(base) + r"(?:s|es)?"

        # possessive (straight or curly); allow plural possessive too
        core += r"(?:'s|’s|s'|s’)?"
        return core

    def boundary_pattern(self, phrase: str) -> re.Pattern:
        """
        Pattern for full phrase.
        - Splits into tokens by spaces.
        - Applies plural/possessive to the *last* token only.
        """
        p = self.safe_phrase_preserve_case(phrase)
        if not p:
            return re.compile(r"a^")

        parts = p.split()
        if not parts:
            return re.compile(r"a^")

        # allow flexible whitespace between tokens
        head_tokens = parts[:-1]
        last_token = parts[-1]

        head_pat = ""
        if head_tokens:
            head_pat = r"\s+".join(re.escape(x.lower()) for x in head_tokens) + r"\s+"

        last_pat = self._token_plural_possessive_pattern(last_token)

        full = rf"{self.boundary_left}{head_pat}{last_pat}{self.boundary_right}"
        return re.compile(full, flags=re.IGNORECASE)

    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        t = self.normalize_one_line(text)
        if not t:
            return False

        w1 = self.safe_phrase_preserve_case(word1)
        w2 = self.safe_phrase_preserve_case(word2)
        if not w1 or not w2:
            return False

        return (self.boundary_pattern(w1).search(t) is not None) and (
            self.boundary_pattern(w2).search(t) is not None
        )


