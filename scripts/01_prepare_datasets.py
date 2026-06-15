from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, RAW_DIR, RANDOM_STATE  # noqa: E402
from src.data.load_data import find_first_table, read_table  # noqa: E402
from src.data.preprocess_arabic import clean_arabic_dataframe  # noqa: E402
from src.data.preprocess_english import clean_english_dataframe, normalize_whitespace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SemEval, RAID, and Arabic clean datasets.")
    parser.add_argument("--semeval-file", default=None, help="Optional raw SemEval file path.")
    parser.add_argument("--raid-file", default=None, help="Optional raw RAID file path.")
    parser.add_argument("--arabic-file", default=None, help="Optional raw Arabic POC file path.")
    parser.add_argument("--raid-max-human", type=int, default=5000)
    parser.add_argument("--raid-max-ai", type=int, default=5000)
    parser.add_argument("--raid-max-scan", type=int, default=250000)
    parser.add_argument("--skip-raid-stream", action="store_true")
    return parser.parse_args()


def prepare_semeval(args: argparse.Namespace) -> None:
    path = Path(args.semeval_file) if args.semeval_file else find_first_table(RAW_DIR / "semeval")
    if path is None:
        print("SemEval raw file not found. Put CSV/JSONL/Parquet under data/raw/semeval/ or pass --semeval-file.")
        return
    raw = read_table(path)
    clean = clean_english_dataframe(raw, dataset_name="semeval")
    clean.to_csv(PROCESSED_DIR / "semeval_english_clean.csv", index=False)
    print(f"Saved SemEval clean data: {clean.shape}")


def prepare_raid_from_file(path: Path) -> None:
    raw = read_table(path)
    metadata = [col for col in ["domain", "generator", "model", "attack", "decoding"] if col in raw.columns]
    clean = clean_english_dataframe(raw, dataset_name="raid", text_col="text" if "text" in raw.columns else None, label_col="label" if "label" in raw.columns else None, metadata_cols=metadata)
    if "model" in clean.columns and "generator" not in clean.columns:
        clean = clean.rename(columns={"model": "generator"})
    for col in ["domain", "generator", "attack", "decoding"]:
        if col not in clean.columns:
            clean[col] = "unknown"
    clean = clean[["text", "label", "dataset_name", "domain", "generator", "attack", "decoding"]]
    clean.to_csv(PROCESSED_DIR / "raid_english_clean.csv", index=False)
    print(f"Saved RAID clean data from file: {clean.shape}")


def prepare_raid_stream(args: argparse.Namespace) -> None:
    print("Streaming RAID from Hugging Face. This may take time.")
    dataset = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    rows = []
    seen = set()
    human_count = 0
    ai_count = 0
    for idx, row in enumerate(dataset, start=1):
        if idx > args.raid_max_scan:
            break
        text = normalize_whitespace(row.get("generation", ""))
        if len(text.split()) < 30 or text in seen:
            continue
        model = str(row.get("model", "") or "")
        label = 0 if model.lower() == "human" else 1
        if label == 0 and human_count >= args.raid_max_human:
            continue
        if label == 1 and ai_count >= args.raid_max_ai:
            continue
        seen.add(text)
        human_count += int(label == 0)
        ai_count += int(label == 1)
        rows.append(
            {
                "text": text,
                "label": label,
                "dataset_name": "raid",
                "domain": row.get("domain", "unknown"),
                "generator": model,
                "attack": row.get("attack", "none"),
                "decoding": row.get("decoding", "unknown"),
            }
        )
        if len(rows) % 1000 == 0:
            print(f"RAID kept {len(rows)} rows after scanning {idx}; human={human_count}, ai={ai_count}")
        if human_count >= args.raid_max_human and ai_count >= args.raid_max_ai:
            break
    clean = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    clean.to_csv(PROCESSED_DIR / "raid_english_clean.csv", index=False)
    print(f"Saved RAID clean data from stream: {clean.shape}")


def prepare_arabic(args: argparse.Namespace) -> None:
    path = Path(args.arabic_file) if args.arabic_file else find_first_table(RAW_DIR / "arabic")
    if path is None:
        print("Arabic raw file not found. Skipping Arabic POC preparation.")
        return
    raw = read_table(path)
    clean = clean_arabic_dataframe(raw, dataset_name="arabic")
    clean.to_csv(PROCESSED_DIR / "arabic_clean.csv", index=False)
    print(f"Saved Arabic clean data: {clean.shape}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    prepare_semeval(args)
    raid_path = Path(args.raid_file) if args.raid_file else find_first_table(RAW_DIR / "raid")
    if raid_path is not None:
        prepare_raid_from_file(raid_path)
    elif not args.skip_raid_stream:
        prepare_raid_stream(args)
    else:
        print("RAID raw file not found and streaming disabled.")
    prepare_arabic(args)


if __name__ == "__main__":
    main()
