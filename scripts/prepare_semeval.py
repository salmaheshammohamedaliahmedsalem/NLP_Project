from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from src.data.load_data import read_table  # noqa: E402
from src.data.preprocess_english import normalize_whitespace  # noqa: E402


TEXT_CANDIDATES = ("text", "content", "generation", "document", "article", "response", "answer")
LABEL_CANDIDATES = ("label", "labels", "class", "generated", "is_ai", "is_generated")
TABLE_PATTERNS = ("*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet", "*.pq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SemEval English binary AI-vs-human data.")
    parser.add_argument("--input", type=Path, default=None, help="Optional file or directory under data/raw/semeval.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "semeval_english_clean.csv")
    return parser.parse_args()


def discover_tables(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in TABLE_PATTERNS:
        files.extend(path.rglob(pattern))
    return sorted(set(files))


def detect_column(df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(f"No {kind} column found. Columns: {list(df.columns)}")


def map_semeval_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("missing label")
    normalized = str(value).strip().lower()
    human_values = {"0", "human", "humans", "real", "authentic", "student", "not_generated", "not generated"}
    ai_values = {"1", "ai", "machine", "generated", "llm", "model", "chatgpt", "gpt", "synthetic"}
    if normalized in human_values:
        return 0
    if normalized in ai_values:
        return 1
    raise ValueError(f"Unknown SemEval label value: {value!r}")


def clean_one_file(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_table(path)
    text_col = detect_column(raw, TEXT_CANDIDATES, "text")
    label_col = detect_column(raw, LABEL_CANDIDATES, "label")
    unique_labels = sorted(raw[label_col].dropna().astype(str).unique().tolist())[:50]
    print(f"\nSemEval file: {path}")
    print(f"Columns: {list(raw.columns)}")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")
    print(f"Unique labels: {unique_labels}")

    clean = raw[[text_col, label_col]].copy()
    rows_before = len(clean)
    missing_text = int(clean[text_col].isna().sum())
    missing_label = int(clean[label_col].isna().sum())
    clean = clean.dropna(subset=[text_col, label_col])
    clean["text"] = clean[text_col].map(normalize_whitespace)
    clean = clean[clean["text"].str.len() > 0].copy()
    clean["label"] = clean[label_col].map(map_semeval_label).astype(int)
    clean["dataset_name"] = "semeval_english"
    clean["source_file"] = str(path.relative_to(ROOT))
    clean = clean[["text", "label", "dataset_name", "source_file"]]

    duplicate_count = int(clean.duplicated(subset=["text"]).sum())
    clean = clean.drop_duplicates(subset=["text"]).reset_index(drop=True)
    summary = {
        "file": str(path.relative_to(ROOT)),
        "rows_before": rows_before,
        "missing_text": missing_text,
        "missing_label": missing_label,
        "duplicates_removed": duplicate_count,
        "rows_after": len(clean),
        "class_distribution": clean["label"].value_counts().sort_index().to_dict(),
    }
    return clean, summary


def write_summary(summaries: list[dict[str, object]], clean: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SemEval Dataset Preparation Summary",
        "",
        "Labels are mapped only from explicit binary human/AI values. Unknown labels raise an error.",
        "",
        f"- Final rows: {len(clean)}",
        f"- Final class distribution: {clean['label'].value_counts().sort_index().to_dict() if not clean.empty else {}}",
        f"- Final columns: {list(clean.columns)}",
        "",
        "## Source Files",
        "",
    ]
    for item in summaries:
        lines.extend(
            [
                f"### {item['file']}",
                f"- Rows before cleaning: {item['rows_before']}",
                f"- Missing text: {item['missing_text']}",
                f"- Missing labels: {item['missing_label']}",
                f"- Duplicate texts removed: {item['duplicates_removed']}",
                f"- Rows after cleaning: {item['rows_after']}",
                f"- Class distribution: {item['class_distribution']}",
                "",
            ]
        )
    (REPORTS_DIR / "semeval_dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_path = args.input or (ROOT / "data" / "raw" / "semeval")
    tables = discover_tables(raw_path)
    if not tables:
        raise FileNotFoundError("No SemEval tables found. Put raw files under data/raw/semeval/ or pass --input.")

    cleaned: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for table in tables:
        try:
            clean, summary = clean_one_file(table)
        except Exception as exc:
            print(f"Skipping {table}: {type(exc).__name__}: {exc}")
            continue
        if not clean.empty:
            cleaned.append(clean)
            summaries.append(summary)

    if not cleaned:
        raise RuntimeError("No SemEval files could be prepared. Check columns and label values printed above.")

    final = pd.concat(cleaned, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    write_summary(summaries, final)
    print(f"\nSaved SemEval clean data: {args.output} shape={final.shape}")


if __name__ == "__main__":
    main()
