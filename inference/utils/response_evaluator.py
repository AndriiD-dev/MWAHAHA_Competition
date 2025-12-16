from __future__ import annotations

import random
import re

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from inference.utils.spacy_extractor import RequiredWordsChecker
from inference.config import ResponseEvaluatorConfig

from transformers import RobertaTokenizerFast, AutoModelForSequenceClassification
import torch



@dataclass
class ResponseEvaluator:
    rng: random.Random = field(default_factory=random.Random)
    config: ResponseEvaluatorConfig = ResponseEvaluatorConfig
    checker: Optional[RequiredWordsChecker] = None
    CLASSIFIER_MODEL_ID = "Humor-Research/humor-detection-comb-23"
    classifier_model = None
    classifier_tokenizer = None

    def __post_init__(self) -> None:
        if self.checker is None:
            self.checker = RequiredWordsChecker(settings=self.config.required_words)

        if self.classifier_tokenizer is None:
            self.classifier_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        if self.classifier_model is None:    
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.CLASSIFIER_MODEL_ID)

        
    def required_words_present(self, text: str, word1: str, word2: str) -> bool:
        assert self.checker is not None
        return self.checker.required_words_present(text, word1, word2)
    
    def is_humorous(self, text: str):
        inputs = self.classifier_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
        print(outputs)
        logits = outputs.logits  # shape [1, 2]
        probs = torch.softmax(logits, dim=-1).squeeze()
        # Assuming label 1 = humorous, 0 = non-humorous (you can swap if needed)
        _, p_humor = probs.tolist()
        return True if p_humor > 0.5 else False
    
    def is_good(self, text: str, word1: str, word2: str) -> bool:
        return self.required_words_present(text, word1, word2) and self.is_humorous(text)
        