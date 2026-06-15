from __future__ import annotations

import re

import pandas as pd

from src.data.preprocess_english import clean_english_dataframe


DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = "\u0640"


def normalize_arabic_text(text: object) -> str:
    value = str(text or "")
    value = DIACRITICS_RE.sub("", value)
    value = value.replace(TATWEEL, "")
    value = re.sub("[إأآا]", "ا", value)
    value = value.replace("ى", "ي")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_arabic_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "arabic",
    text_col: str | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    clean = clean_english_dataframe(df, dataset_name=dataset_name, text_col=text_col, label_col=label_col)
    clean["text"] = clean["text"].map(normalize_arabic_text)
    clean = clean[clean["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return clean[["text", "label", "dataset_name"]]
