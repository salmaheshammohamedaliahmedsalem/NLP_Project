from __future__ import annotations

import re

import pandas as pd


TEXT_CANDIDATES = ("text", "content", "generation", "document", "article", "response")
LABEL_CANDIDATES = ("label", "labels", "generated", "is_ai", "class", "model")


def normalize_whitespace(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def detect_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_CANDIDATES:
        if column in df.columns:
            return column
    object_cols = [column for column in df.columns if df[column].dtype == "object"]
    if not object_cols:
        raise ValueError(f"No text-like column found. Columns: {list(df.columns)}")
    return max(object_cols, key=lambda col: df[col].astype(str).str.len().mean())


def detect_label_column(df: pd.DataFrame) -> str:
    for column in LABEL_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No label-like column found. Columns: {list(df.columns)}")


def encode_binary_label(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"0", "human", "real", "false", "not_generated", "not generated"}:
        return 0
    if normalized in {"1", "ai", "machine", "generated", "true", "llm"}:
        return 1
    if "human" in normalized:
        return 0
    return 1


def clean_english_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str | None = None,
    label_col: str | None = None,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    text_col = text_col or detect_text_column(df)
    label_col = label_col or detect_label_column(df)
    metadata_cols = metadata_cols or []

    keep_cols = [text_col, label_col] + [col for col in metadata_cols if col in df.columns]
    clean = df[keep_cols].copy()
    clean = clean.dropna(subset=[text_col, label_col])
    clean["text"] = clean[text_col].map(normalize_whitespace)
    clean = clean[clean["text"].str.len() > 0].copy()
    clean["label"] = clean[label_col].map(encode_binary_label).astype(int)
    clean["dataset_name"] = dataset_name
    clean = clean.drop_duplicates(subset=["text"]).reset_index(drop=True)

    output_cols = ["text", "label", "dataset_name"]
    for col in metadata_cols:
        if col in clean.columns and col not in output_cols:
            output_cols.append(col)
    return clean[output_cols]
