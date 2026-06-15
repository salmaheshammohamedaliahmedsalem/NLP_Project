from __future__ import annotations

import pandas as pd

from src.config import MODEL_NAMES
from src.models.model_factory import make_model


def train_model(model_name: str, train_df: pd.DataFrame):
    model = make_model(model_name)
    model.fit(train_df["text"], train_df["label"])
    return model


def train_all_models(train_df: pd.DataFrame, model_names: tuple[str, ...] = MODEL_NAMES):
    return {model_name: train_model(model_name, train_df) for model_name in model_names}
