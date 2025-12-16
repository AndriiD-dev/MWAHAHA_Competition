from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import torch
from transformers import AutoModelForSequenceClassification, RobertaTokenizerFast

from inference.config import ResponseEvaluatorConfig
from inference.utils.spacy_extractor import RequiredWordsChecker


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

    def is_humorous(self, text: str) -> bool:
        assert self.__class__.classifier_tokenizer is not None
        assert self.__class__.classifier_model is not None

        tokenizer = self.__class__.classifier_tokenizer
        model = self.__class__.classifier_model

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        if next(model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits  # shape [1, 2]
        probs = torch.softmax(logits, dim=-1).squeeze()
        _, p_humor = probs.tolist()
        return p_humor > 0.5

    def is_good(self, text: str, word1: str, word2: str) -> bool:
        return self.required_words_present(text, word1, word2) and self.is_humorous(text)
