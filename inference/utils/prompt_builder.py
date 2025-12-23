from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from inference.config import PromptBuilderConfig
from inference.utils.spacy_extractor import SpacyAnchorExtractor
from inference.utils.wiki_reader import WikipediaReader


_WS_RE = re.compile(r"\s+")


def normalize_one_line(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def safe_word(word: str) -> str:
    return normalize_one_line(word).strip()


@dataclass
class PromptBuilder:
    """
    Single source of truth for prompt construction.

    - uses WikipediaReader for microcard FACTS
    - uses SpacyAnchorExtractor for headline noun selection
    """

    config: PromptBuilderConfig = field(default_factory=PromptBuilderConfig)
    wiki: WikipediaReader | None = None
    spacy: SpacyAnchorExtractor | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.wiki is None:
            self.wiki = WikipediaReader(settings=self.config.wiki)
            try:
                self.wiki.load_cache(self.config.paths.wiki_cache_path)
            except Exception:
                pass

        if self.spacy is None:
            self.spacy = SpacyAnchorExtractor(
                settings=self.config.spacy,
                generic_nouns=set(x.lower() for x in self.config.generic_nouns),
                extra_stopwords=set(x.lower() for x in self.config.extra_stopwords),
            )

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

    def format_cards_block_unlabeled(self, word1: str, word2: str) -> str:
        """
        Caption task wants the same microcards, but without:
          - "FACTS:"
          - "- <word>:" labels
          - any "required words" framing
        We still include enough concrete text so the caption can naturally reuse it.
        """
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

            # Intentionally *do not* include the word label.
            # The Wikipedia sentence often repeats the term; if not, the caption still has to pass internal checks.
            return "- " + " | ".join(parts)

        return "CARDS:\n" + fmt_one(w1) + "\n" + fmt_one(w2)

    def save_wiki_cache(self) -> None:
        assert self.wiki is not None
        try:
            self.wiki.save_cache(self.config.paths.wiki_cache_path)
        except Exception:
            pass

   
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



    def build_caption_messages(self, scene: str, noun1: str, noun2: str) -> List[Dict[str, str]]:
        s = normalize_one_line(scene)
        n1 = safe_word(noun1)
        n2 = safe_word(noun2)

        system = self.config.prompts.caption_common
        cards = self.format_cards_block_unlabeled(n1, n2)

        user = normalize_one_line(self.config.prompts.caption_task.format(scene=s, cards=cards))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
