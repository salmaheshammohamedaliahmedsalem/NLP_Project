from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, SPLITS_DIR  # noqa: E402
from src.data.load_data import load_clean_dataset  # noqa: E402
from src.data.make_splits import save_split, stratified_train_test_split  # noqa: E402


def make_split_for(dataset_key: str, filename: str) -> None:
    path = PROCESSED_DIR / filename
    if not path.exists():
        print(f"Skipping {dataset_key}: missing {path}")
        return
    df = load_clean_dataset(path)
    train_df, test_df = stratified_train_test_split(df)
    save_split(train_df, test_df, SPLITS_DIR / dataset_key)
    print(f"Saved {dataset_key} split: train={len(train_df)}, test={len(test_df)}")


def main() -> None:
    make_split_for("semeval", "semeval_english_clean.csv")
    make_split_for("raid", "raid_english_clean.csv")
    make_split_for("arabic", "arabic_clean.csv")


if __name__ == "__main__":
    main()
