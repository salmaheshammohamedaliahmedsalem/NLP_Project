from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")


def find_first_table(directory: str | Path) -> Path | None:
    directory = Path(directory)
    for pattern in ("*.csv", "*.jsonl", "*.json", "*.parquet", "*.tsv"):
        files = sorted(directory.glob(pattern))
        if files:
            return files[0]
    return None


def load_clean_dataset(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    required = {"text", "label", "dataset_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clean dataset missing required columns {missing}: {path}")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)
