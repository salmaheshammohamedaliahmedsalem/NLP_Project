from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402
from src.data.download_log import append_download_log  # noqa: E402
from src.data.load_data import read_table  # noqa: E402
from src.data.preprocess_arabic import normalize_arabic_text  # noqa: E402


ABSTRACT_DATASET = "KFUPM-JRCAI/arabic-generated-abstracts"
SOCIAL_DATASET = "KFUPM-JRCAI/arabic-generated-social-media-posts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Arabic generated text proof-of-concept data.")
    parser.add_argument("--input", type=Path, default=None, help="Optional local Arabic raw file.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "arabic_poc_clean.csv")
    parser.add_argument("--include-social", action="store_true", help="Also try KFUPM generated social-media posts.")
    parser.add_argument("--skip-hf", action="store_true", help="Only use local files under data/raw/arabic.")
    return parser.parse_args()


def save_hf_dataset(dataset_name: str, raw_dir: Path) -> list[Path]:
    from datasets import load_dataset

    saved: list[Path] = []
    dataset_dict = load_dataset(dataset_name)
    safe_name = dataset_name.split("/")[-1].replace("-", "_")
    for split, table in dataset_dict.items():
        df = table.to_pandas()
        path = raw_dir / f"{safe_name}_{split}.parquet"
        df.to_parquet(path, index=False)
        saved.append(path)
        print(f"Saved raw Arabic split: {path} shape={df.shape}")
    append_download_log(
        dataset_name=dataset_name,
        source=f"Hugging Face dataset {dataset_name}",
        command=f"datasets.load_dataset('{dataset_name}')",
        success=True,
        local_raw_path=raw_dir,
        notes=f"Saved {len(saved)} parquet files.",
    )
    return saved


def discover_or_download(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [args.input]
    raw_dir = RAW_DIR / "arabic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(
        path
        for pattern in ("*.csv", "*.tsv", "*.jsonl", "*.json", "*.parquet", "*.pq")
        for path in raw_dir.rglob(pattern)
    )
    if existing_files:
        return existing_files
    if args.skip_hf:
        return []

    files: list[Path] = []
    for dataset_name in [ABSTRACT_DATASET] + ([SOCIAL_DATASET] if args.include_social else []):
        try:
            files.extend(save_hf_dataset(dataset_name, raw_dir))
        except Exception as exc:
            append_download_log(
                dataset_name=dataset_name,
                source=f"Hugging Face dataset {dataset_name}",
                command=f"datasets.load_dataset('{dataset_name}')",
                success=False,
                local_raw_path=raw_dir,
                notes=f"{type(exc).__name__}: {exc}",
            )
            print(f"Arabic download failed for {dataset_name}: {type(exc).__name__}: {exc}")
    return files


def generated_label_from_column(column: str) -> tuple[int, str] | None:
    normalized = column.lower()
    if normalized.startswith("original_") or normalized in {"original", "human", "human_text"}:
        return 0, "human"
    if "generated" in normalized:
        generator = normalized.replace("_generated_abstract", "").replace("_generated_post", "")
        generator = generator.replace("generated_", "").replace("_generated", "")
        return 1, generator or "unknown_ai"
    return None


def infer_domain(path: Path, column: str) -> str:
    value = f"{path.name} {column}".lower()
    if "post" in value or "social" in value:
        return "social_media"
    if "abstract" in value:
        return "academic_abstract"
    return "arabic_text"


def wide_to_long(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    print(f"\nArabic file: {path}")
    print(f"Columns: {list(raw.columns)}")
    rows: list[pd.DataFrame] = []
    for column in raw.columns:
        mapping = generated_label_from_column(column)
        if mapping is None:
            continue
        label, generator = mapping
        part = pd.DataFrame(
            {
                "text": raw[column].map(normalize_arabic_text),
                "label": label,
                "dataset_name": "arabic_poc",
                "domain": infer_domain(path, column),
                "generator": generator,
                "source_column": column,
                "source_file": str(path.relative_to(ROOT)),
            }
        )
        rows.append(part)

    if not rows:
        raise ValueError(
            "No Arabic original/generated columns found. Expected columns like original_abstract, "
            "openai_generated_abstract, original_post, or *_generated_post."
        )

    clean = pd.concat(rows, ignore_index=True)
    clean = clean.dropna(subset=["text"])
    clean = clean[clean["text"].str.len() > 0]
    clean = clean.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Arabic class distribution from {path.name}: {clean['label'].value_counts().sort_index().to_dict()}")
    return clean


def write_summary(clean: pd.DataFrame, output: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Arabic POC Dataset Preparation Summary",
        "",
        "Arabic text is lightly normalized: diacritics and tatweel are removed, Alef variants are normalized,",
        "`ى` is normalized to `ي`, and whitespace is collapsed. Punctuation and casing are otherwise preserved.",
        "",
        f"- Final shape: {clean.shape}",
        f"- Output: `{output}`",
        f"- Class distribution: {clean['label'].value_counts().sort_index().to_dict()}",
        f"- Domains: {clean['domain'].value_counts().to_dict()}",
        f"- Generators: {clean['generator'].value_counts().to_dict()}",
        f"- Source columns: {clean['source_column'].value_counts().to_dict()}",
        f"- Columns: {list(clean.columns)}",
        "",
    ]
    (REPORTS_DIR / "arabic_dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = discover_or_download(args)
    if not files:
        raise FileNotFoundError(
            "No Arabic raw files found. Provide --input, place files under data/raw/arabic/, "
            "or run without --skip-hf to download KFUPM-JRCAI data."
        )

    cleaned: list[pd.DataFrame] = []
    for path in files:
        raw = read_table(path)
        cleaned.append(wide_to_long(raw, path))

    final = pd.concat(cleaned, ignore_index=True).drop_duplicates(subset=["text"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    write_summary(final, args.output)
    print(f"\nSaved Arabic POC clean data: {args.output} shape={final.shape}")


if __name__ == "__main__":
    main()
