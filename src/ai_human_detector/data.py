from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


LEAKY_COLUMNS = {
    "id",
    "prompt",
    "source",
    "ai_model",
    "label",
    "word_count",
    "char_count",
    "complexity_score",
    "is_multiline_code",
    "language",
}


AI_MARKER_RE = re.compile(
    r"(\(?\bAI[- ]generated\b\)?|#\s*AI[- ]generated|//\s*AI[- ]generated)",
    flags=re.IGNORECASE,
)


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    text_candidates = ["content", "text", "document", "article", "response"]
    label_candidates = ["label", "generated", "is_ai", "class"]
    domain_candidates = ["type", "domain", "category", "topic"]

    text_col = next((col for col in text_candidates if col in df.columns), None)
    label_col = next((col for col in label_candidates if col in df.columns), None)
    domain_col = next((col for col in domain_candidates if col in df.columns), None)

    if text_col is None:
        object_cols = [col for col in df.columns if df[col].dtype == "object"]
        text_col = max(object_cols, key=lambda col: df[col].astype(str).str.len().mean(), default=None)

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Found columns: {list(df.columns)}")

    return text_col, label_col, domain_col


def encode_label(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"human", "0", "false", "no", "human-written", "real"}:
        return 0
    if normalized in {"ai", "1", "true", "yes", "generated", "machine"}:
        return 1
    return 0 if "human" in normalized else 1


def strip_ai_markers(text: object) -> str:
    stripped = AI_MARKER_RE.sub(" ", str(text))
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_for_dedup(text: object) -> str:
    normalized = str(text).lower()
    normalized = AI_MARKER_RE.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9_{}()\[\];:.,+\-*/=<>#'\" ]+", "", normalized)
    return normalized.strip()


def load_raw_dataset(path: str | Path) -> tuple[pd.DataFrame, str, str, str | None]:
    df = pd.read_csv(path)
    text_col, label_col, domain_col = detect_columns(df)
    return df, text_col, label_col, domain_col


def clean_dataset(df: pd.DataFrame, text_col: str, label_col: str, domain_col: str | None) -> tuple[pd.DataFrame, dict]:
    audit: dict[str, object] = {
        "raw_rows": len(df),
        "raw_columns": list(df.columns),
        "text_col": text_col,
        "label_col": label_col,
        "domain_col": domain_col,
    }

    working = df.dropna(subset=[text_col, label_col]).copy()
    if "prompt" in working.columns:
        working["prompt_group"] = working["prompt"].fillna("missing").astype(str)
    audit["rows_after_required_not_null"] = len(working)
    audit["ai_marker_rows_before_cleaning"] = int(
        working[text_col].astype(str).str.contains("AI-generated", case=False, na=False).sum()
    )

    working["content"] = working[text_col].apply(strip_ai_markers)
    working["label_binary"] = working[label_col].apply(encode_label).astype(int)
    if domain_col and domain_col in working.columns:
        working["domain"] = working[domain_col].fillna("unknown").astype(str)
    else:
        working["domain"] = "unknown"

    working = working[working["content"].str.len() > 0].copy()
    working["content_norm"] = working["content"].apply(normalize_for_dedup)

    label_counts_per_text = working.groupby("content_norm")["label_binary"].nunique()
    conflicting_norms = label_counts_per_text[label_counts_per_text > 1].index
    audit["conflicting_normalized_texts"] = int(len(conflicting_norms))
    if len(conflicting_norms) > 0:
        working = working[~working["content_norm"].isin(conflicting_norms)].copy()

    audit["duplicate_normalized_rows_before_drop"] = int(working.duplicated("content_norm", keep=False).sum())
    working = working.drop_duplicates(subset=["content_norm"]).reset_index(drop=True)
    audit["clean_rows"] = len(working)
    audit["removed_rows_total"] = int(audit["raw_rows"] - len(working))
    audit["label_distribution_clean"] = working["label_binary"].value_counts().sort_index().to_dict()
    audit["domain_distribution_clean"] = working["domain"].value_counts().to_dict()

    safe_cols = ["content", "label_binary", "domain", "content_norm"]
    if "prompt_group" in working.columns:
        safe_cols.append("prompt_group")

    return working[safe_cols].copy(), audit


def metadata_leakage_report(raw_df: pd.DataFrame, label_col: str, output_path: str | Path) -> pd.DataFrame:
    rows = []
    if label_col not in raw_df.columns:
        return pd.DataFrame()

    labels = raw_df[label_col].apply(encode_label)
    for col in raw_df.columns:
        if col == label_col:
            continue
        series = raw_df[col].fillna("MISSING").astype(str)
        if series.nunique(dropna=False) > 80:
            continue
        table = pd.crosstab(series, labels, normalize="index")
        purity = float(table.max(axis=1).mean()) if len(table) else 0.0
        rows.append(
            {
                "column": col,
                "unique_values": int(series.nunique(dropna=False)),
                "mean_label_purity": purity,
                "risk": "high" if purity >= 0.95 else "medium" if purity >= 0.80 else "low",
            }
        )

    report = pd.DataFrame(rows).sort_values(["mean_label_purity", "unique_values"], ascending=[False, True])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def make_random_split(df: pd.DataFrame, random_state: int = 42) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["label_binary"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label_binary"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def make_prompt_grouped_split(df: pd.DataFrame, random_state: int = 42) -> dict[str, pd.DataFrame] | None:
    if "prompt_group" not in df.columns or df["prompt_group"].nunique() < 3:
        return None

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, df["label_binary"], groups=df["prompt_group"]))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_size = 0.20
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    tr_idx, val_idx = next(inner.split(train_val, train_val["label_binary"], groups=train_val["prompt_group"]))
    return {
        "train": train_val.iloc[tr_idx].reset_index(drop=True),
        "val": train_val.iloc[val_idx].reset_index(drop=True),
        "test": test_df,
    }


def make_domain_holdout_splits(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    splits = {}
    domains = [domain for domain, count in df["domain"].value_counts().items() if count >= 50]
    for domain in domains:
        train_val = df[df["domain"] != domain].copy()
        test_df = df[df["domain"] == domain].copy()
        if train_val["label_binary"].nunique() < 2 or test_df["label_binary"].nunique() < 2:
            continue
        try:
            train_df, val_df = train_test_split(
                train_val,
                test_size=0.20,
                random_state=42,
                stratify=train_val["label_binary"],
            )
        except ValueError:
            continue
        splits[f"holdout_{domain}"] = {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }
    return splits
