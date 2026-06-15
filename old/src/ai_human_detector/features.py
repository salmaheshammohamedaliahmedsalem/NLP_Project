from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]*")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def extract_one(text: object) -> dict[str, float]:
    value = str(text)
    chars = len(value)
    words = WORD_RE.findall(value)
    lower_words = [word.lower() for word in words]
    word_count = len(words)
    unique_words = len(set(lower_words))
    sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(value) if sentence.strip()]
    sentence_lengths = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    word_lengths = [len(word) for word in words]
    lines = value.splitlines() or [value]

    punctuation_count = sum(1 for char in value if char in ".,;:!?")
    digit_count = sum(char.isdigit() for char in value)
    upper_count = sum(char.isupper() for char in value)
    alpha_count = sum(char.isalpha() for char in value)
    whitespace_count = sum(char.isspace() for char in value)

    bigrams = list(zip(lower_words, lower_words[1:]))
    trigrams = list(zip(lower_words, lower_words[1:], lower_words[2:]))

    return {
        "char_count": float(chars),
        "word_count": float(word_count),
        "sentence_count": float(len(sentences)),
        "line_count": float(len(lines)),
        "avg_word_length": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "std_word_length": float(np.std(word_lengths)) if word_lengths else 0.0,
        "avg_sentence_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "std_sentence_length": float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
        "lexical_diversity": _safe_divide(unique_words, word_count),
        "hapax_ratio": _safe_divide(sum(1 for count in Counter(lower_words).values() if count == 1), word_count),
        "word_entropy": _entropy(lower_words),
        "punctuation_ratio": _safe_divide(punctuation_count, chars),
        "comma_ratio": _safe_divide(value.count(","), chars),
        "period_ratio": _safe_divide(value.count("."), chars),
        "question_ratio": _safe_divide(value.count("?"), chars),
        "exclamation_ratio": _safe_divide(value.count("!"), chars),
        "digit_ratio": _safe_divide(digit_count, chars),
        "uppercase_ratio": _safe_divide(upper_count, alpha_count),
        "whitespace_ratio": _safe_divide(whitespace_count, chars),
        "quote_count": float(value.count('"') + value.count("'")),
        "semicolon_count": float(value.count(";")),
        "brace_count": float(sum(value.count(char) for char in "{}[]()")),
        "operator_count": float(sum(value.count(op) for op in ["=", "+", "-", "*", "/", "<", ">"])),
        "repeated_word_ratio": _safe_divide(word_count - unique_words, word_count),
        "repeated_bigram_ratio": _safe_divide(len(bigrams) - len(set(bigrams)), len(bigrams)),
        "repeated_trigram_ratio": _safe_divide(len(trigrams) - len(set(trigrams)), len(trigrams)),
        "avg_line_length": float(np.mean([len(line) for line in lines])) if lines else 0.0,
        "max_line_length": float(max([len(line) for line in lines])) if lines else 0.0,
    }


def extract_stylometric_frame(texts: pd.Series | list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([extract_one(text) for text in texts])
    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
