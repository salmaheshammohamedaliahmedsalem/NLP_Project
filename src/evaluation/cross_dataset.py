from __future__ import annotations

import pandas as pd

from src.config import MODEL_NAMES
from src.evaluation.evaluate import evaluate_model
from src.models.train_models import train_model


def run_cross_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, train_name: str, test_name: str) -> list[dict]:
    rows = []
    for model_name in MODEL_NAMES:
        model = train_model(model_name, train_df)
        rows.append(evaluate_model(model, test_df, model_name, train_name, test_name))
    return rows
