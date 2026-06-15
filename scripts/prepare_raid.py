from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from src.data.load_data import find_first_table, read_table  # noqa: E402
from src.data.preprocess_english import normalize_whitespace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced English RAID subset.")
    parser.add_argument("--input", type=Path, default=None, help="Optional RAID raw file.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "raid_english_clean.csv")
    parser.add_argument("--max-human", type=int, default=10_000)
    parser.add_argument("--max-ai", type=int, default=10_000)
    parser.add_argument("--max-scan", type=int, default=500_000)
    parser.add_argument("--allow-hf-fallback", action="store_true", help="Stream from Hugging Face if no local raw file exists.")
    return parser.parse_args()


def detect_text_column(df: pd.DataFrame) -> str:
    for column in ("generation", "text", "content", "article", "response", "document"):
        if column in df.columns:
            return column
    raise ValueError(f"No RAID text column found. Columns: {list(df.columns)}")


def map_raid_label(row: pd.Series) -> int:
    if "model" in row.index and pd.notna(row["model"]):
        return 0 if str(row["model"]).strip().lower() == "human" else 1
    for column in ("label", "is_ai", "generated", "class"):
        if column in row.index and pd.notna(row[column]):
            normalized = str(row[column]).strip().lower()
            if normalized in {"0", "human", "real", "false", "not_generated", "not generated"}:
                return 0
            if normalized in {"1", "ai", "machine", "generated", "true", "llm"}:
                return 1
            raise ValueError(f"Unknown RAID label value in {column}: {row[column]!r}")
    raise ValueError("RAID requires either model or explicit binary label column.")


def filter_english_if_available(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("language", "lang"):
        if column in df.columns:
            keep = df[column].astype(str).str.lower().isin({"en", "eng", "english"})
            return df[keep].copy()
    return df


def clean_raid_dataframe(raw: pd.DataFrame, max_human: int, max_ai: int) -> pd.DataFrame:
    print(f"RAID columns: {list(raw.columns)}")
    if "model" in raw.columns:
        print(f"RAID unique models sample: {sorted(raw['model'].dropna().astype(str).unique().tolist())[:30]}")
    for label_col in ("label", "is_ai", "generated", "class"):
        if label_col in raw.columns:
            print(f"RAID unique labels in {label_col}: {sorted(raw[label_col].dropna().astype(str).unique().tolist())[:30]}")

    raw = filter_english_if_available(raw)
    text_col = detect_text_column(raw)
    clean = pd.DataFrame()
    clean["text"] = raw[text_col].map(normalize_whitespace)
    clean["label"] = raw.apply(map_raid_label, axis=1).astype(int)
    clean["dataset_name"] = "raid_english"
    clean["domain"] = raw["domain"] if "domain" in raw.columns else "unknown"
    clean["generator"] = raw["model"] if "model" in raw.columns else raw.get("generator", "unknown")
    clean["attack"] = raw["attack"] if "attack" in raw.columns else "none"
    clean["decoding"] = raw["decoding"] if "decoding" in raw.columns else "unknown"
    clean = clean[clean["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)

    human = clean[clean["label"] == 0]
    ai = clean[clean["label"] == 1]
    n_human = min(max_human, len(human))
    n_ai = min(max_ai, len(ai))
    n_balanced = min(n_human, n_ai)
    if n_balanced == 0:
        raise ValueError(f"RAID needs both classes after cleaning. Distribution: {clean['label'].value_counts().to_dict()}")
    balanced = pd.concat(
        [
            human.sample(n=n_balanced, random_state=RANDOM_STATE),
            ai.sample(n=n_balanced, random_state=RANDOM_STATE),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE)
    return balanced.reset_index(drop=True)


def stream_hf_rows(max_scan: int) -> pd.DataFrame:
    from datasets import load_dataset

    rows: list[dict[str, object]] = []
    dataset: Iterable[dict[str, object]] = load_dataset("liamdugan/raid", split="train", streaming=True)
    for idx, row in enumerate(dataset, start=1):
        if idx > max_scan:
            break
        rows.append(row)
        if idx % 25_000 == 0:
            print(f"Scanned {idx} RAID rows from Hugging Face...")
    return pd.DataFrame(rows)


def write_summary(clean: pd.DataFrame, raw_shape: tuple[int, int], output: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAID Dataset Preparation Summary",
        "",
        "RAID labels are derived from `model == human` when available; all other models are AI-generated.",
        "Metadata is preserved for analysis but must not be used as model input.",
        "",
        f"- Raw shape: {raw_shape}",
        f"- Clean balanced shape: {clean.shape}",
        f"- Output: `{output}`",
        f"- Class distribution: {clean['label'].value_counts().sort_index().to_dict()}",
        f"- Domains: {clean['domain'].value_counts().head(20).to_dict()}",
        f"- Generators: {clean['generator'].value_counts().head(20).to_dict()}",
        f"- Attacks: {clean['attack'].value_counts().head(20).to_dict()}",
        f"- Decoding values: {clean['decoding'].value_counts().head(20).to_dict()}",
        f"- Columns: {list(clean.columns)}",
        "",
    ]
    (REPORTS_DIR / "raid_dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input or find_first_table(ROOT / "data" / "raw" / "raid")
    if input_path is not None:
        raw = read_table(input_path)
        print(f"Loaded local RAID raw file: {input_path} shape={raw.shape}")
    elif args.allow_hf_fallback:
        print("No local RAID file found. Streaming Hugging Face RAID train split.")
        raw = stream_hf_rows(args.max_scan)
    else:
        raise FileNotFoundError(
            "No RAID raw file found. Run scripts/download_datasets.py, place a file under data/raw/raid/, "
            "or pass --allow-hf-fallback."
        )

    clean = clean_raid_dataframe(raw, max_human=args.max_human, max_ai=args.max_ai)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    write_summary(clean, raw.shape, args.output)
    print(f"Saved RAID clean data: {args.output} shape={clean.shape}")


if __name__ == "__main__":
    main()
