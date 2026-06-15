from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.stylometric_features import extract_stylometric_features_one


ROOT = Path(__file__).resolve().parents[1]


def find_model_artifacts() -> list[Path]:
    return sorted((ROOT / "outputs" / "models").glob("**/*.joblib"))


def load_model(path: str | Path):
    return joblib.load(path)


def predict_with_score(model, text: str) -> tuple[int, float | None]:
    prediction = int(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        return prediction, float(model.predict_proba([text])[0, 1])
    if hasattr(model, "decision_function"):
        return prediction, float(model.decision_function([text])[0])
    return prediction, None


def stylometric_summary(text: str) -> pd.DataFrame:
    features = extract_stylometric_features_one(text)
    return pd.DataFrame([{"feature": key, "value": value} for key, value in features.items()])
