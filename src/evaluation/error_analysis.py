from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_errors(model, test_df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(test_df["text"])
    errors = test_df.copy()
    errors["predicted_label"] = predictions
    errors = errors[errors["label"] != errors["predicted_label"]].copy()
    errors = errors.rename(columns={"label": "true_label"})
    errors.to_csv(output_path, index=False)
    return errors
