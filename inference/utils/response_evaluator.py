from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, RobertaTokenizerFast

from inference.config import ResponseEvaluatorConfig
from inference.utils.spacy_extractor import RequiredWordsChecker


def _word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


@dataclass
class ResponseEvaluator:
    rng: random.Random = field(default_factory=random.Random)
    config: ResponseEvaluatorConfig = field(default_factory=ResponseEvaluatorConfig)
    checker: Optional[RequiredWordsChecker] = None

    CLASSIFIER_MODEL_ID: ClassVar[str] = "Humor-Research/humor-detection-comb-23"
    classifier_model: ClassVar[Optional[AutoModelForSequenceClassification]] = None
    classifier_tokenizer: ClassVar[Optional[RobertaTokenizerFast]] = None

    def __post_init__(self) -> None:
        if self.checker is None:
            self.checker = RequiredWordsChecker(settings=self.config.required_words)

        if self.__class__.classifier_tokenizer is None:
            self.__class__.classifier_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        if self.__class__.classifier_model is None:
            model = AutoModelForSequenceClassification.from_pretrained(self.CLASSIFIER_MODEL_ID)
            model.eval()
            if torch.cuda.is_available():
                model.to("cuda")
            self.__class__.classifier_model = model

    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        assert self.checker is not None
        return self.checker.required_words_present(text, word1, word2)

    def humor_prob(self, text: str) -> float:
        """
        Returns P(humor) as a float in [0, 1].
        """
        assert self.__class__.classifier_tokenizer is not None
        assert self.__class__.classifier_model is not None

        tokenizer = self.__class__.classifier_tokenizer
        model = self.__class__.classifier_model

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        if next(model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  # [1, 2]
        probs = torch.softmax(logits, dim=-1).squeeze()
        # assuming index 1 corresponds to humor (as used in your original code)
        _p_not, p_humor = probs.tolist()
        return float(p_humor)

    def is_humorous(self, text: str) -> bool:
        return self.humor_prob(text) > 0.5

    def is_short_caption(self, text: str, max_words: int = 20) -> bool:
        return _word_count(text) <= int(max_words)

    def is_good(self, text: str, word1: str, word2: str) -> bool:
        return self.required_words_present(text, word1, word2) and self.is_humorous(text)

    def score_caption_candidate(
        self,
        text: str,
        word1: str,
        word2: str,
        *,
        max_words: int = 20,
    ) -> Tuple[bool, float]:
        """
        Returns: (valid, score)
        valid means it passes:
          - required words present
          - short enough
        Score is humor probability (higher is better).
        """
        t = (text or "").strip()
        if not t:
            return (False, 0.0)

        if not self.required_words_present(t, word1, word2):
            return (False, 0.0)

        if not self.is_short_caption(t, max_words=max_words):
            return (False, 0.0)

        return (True, self.humor_prob(t))
