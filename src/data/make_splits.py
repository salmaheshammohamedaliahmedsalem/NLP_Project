from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def stratified_train_test_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
