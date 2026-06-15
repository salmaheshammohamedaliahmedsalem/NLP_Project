from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!؟?\n]+[.!؟?\n]*", flags=re.UNICODE)
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "of", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "it", "as", "by", "from",
}
AR_STOPWORDS = {"و", "في", "من", "على", "الى", "إلى", "عن", "ان", "أن", "هذا", "هذه", "هو", "هي"}


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def extract_stylometric_features_one(text: object) -> dict[str, float]:
    value = str(text or "")
    chars = len(value)
    words = WORD_RE.findall(value)
    lower_words = [word.lower() for word in words]
    sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(value) if sentence.strip()]
    word_lengths = [len(word) for word in words]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    unique_words = len(set(lower_words))
    counts = Counter(lower_words)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    stopwords = EN_STOPWORDS | AR_STOPWORDS
    stopword_count = sum(1 for word in lower_words if word in stopwords)
    alpha_count = sum(char.isalpha() for char in value)
    uppercase_count = sum(char.isupper() for char in value)
    digit_count = sum(char.isdigit() for char in value)

    return {
        "word_count": float(len(words)),
        "char_count": float(chars),
        "sentence_count": float(len(sentences)),
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "punctuation_count": float(sum(1 for char in value if char in ".,;:!?؟،")),
        "comma_count": float(value.count(",") + value.count("،")),
        "period_count": float(value.count(".")),
        "question_count": float(value.count("?") + value.count("؟")),
        "exclamation_count": float(value.count("!")),
        "uppercase_ratio": safe_divide(uppercase_count, alpha_count),
        "digit_ratio": safe_divide(digit_count, chars),
        "lexical_diversity": safe_divide(unique_words, len(words)),
        "repetition_ratio": safe_divide(repeated, len(words)),
        "stopword_ratio": safe_divide(stopword_count, len(words)),
    }


def extract_stylometric_features(texts) -> pd.DataFrame:
    frame = pd.DataFrame([extract_stylometric_features_one(text) for text in texts])
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
