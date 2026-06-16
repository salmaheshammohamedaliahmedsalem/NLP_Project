from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from src.data.preprocess_english import normalize_whitespace  # noqa: E402


OLD_DATASET_PATH = ROOT / "old" / "ai_vs_human_content_v2_20000.csv"
OUTPUT_PATH = PROCESSED_DIR / "old_ai_clean.csv"


def map_old_label(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized == "human":
        return 0
    if normalized == "ai":
        return 1
    raise ValueError(f"Unknown old dataset label: {value!r}")


def main() -> None:
    if not OLD_DATASET_PATH.exists():
        raise FileNotFoundError(f"Old dataset not found: {OLD_DATASET_PATH}")

    raw = pd.read_csv(OLD_DATASET_PATH)
    print(f"Loaded old dataset: {OLD_DATASET_PATH} shape={raw.shape}")
    print(f"Columns: {list(raw.columns)}")
    print(f"Unique labels: {sorted(raw['label'].dropna().astype(str).unique().tolist())}")

    metadata_cols = [
        "type",
        "source",
        "topic",
        "word_count",
        "char_count",
        "ai_model",
        "language",
        "complexity_score",
        "is_multiline_code",
    ]
    clean = raw[["content", "label"] + [col for col in metadata_cols if col in raw.columns]].copy()
    rows_before = len(clean)
    missing_text = int(clean["content"].isna().sum())
    missing_label = int(clean["label"].isna().sum())
    clean = clean.dropna(subset=["content", "label"])
    clean["text"] = clean["content"].map(normalize_whitespace)
    clean = clean[clean["text"].str.len() > 0].copy()
    clean["label"] = clean["label"].map(map_old_label).astype(int)
    clean["dataset_name"] = "old_ai"
    duplicate_count = int(clean.duplicated(subset=["text"]).sum())
    clean = clean.drop_duplicates(subset=["text"]).reset_index(drop=True)

    output_cols = ["text", "label", "dataset_name"] + [col for col in metadata_cols if col in clean.columns]
    clean = clean[output_cols]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# Old Dataset Preparation Summary",
        "",
        "This benchmark uses the previous project CSV as a separate comparison dataset. It is not mixed with SemEval or RAID.",
        "",
        f"- Source: `{OLD_DATASET_PATH.relative_to(ROOT)}`",
        f"- Rows before cleaning: {rows_before}",
        f"- Missing text: {missing_text}",
        f"- Missing labels: {missing_label}",
        f"- Duplicate texts removed: {duplicate_count}",
        f"- Final rows: {len(clean)}",
        f"- Class distribution: {clean['label'].value_counts().sort_index().to_dict()}",
        f"- Content types: {clean['type'].value_counts().to_dict() if 'type' in clean else {}}",
        f"- Languages: {clean['language'].value_counts().to_dict() if 'language' in clean else {}}",
        f"- Final columns: {list(clean.columns)}",
        "",
    ]
    (REPORTS_DIR / "old_dataset_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved old clean data: {OUTPUT_PATH} shape={clean.shape}")


if __name__ == "__main__":
    main()
