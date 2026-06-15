from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def get_scores(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return None


def evaluate_model(model, test_df: pd.DataFrame, model_name: str, train_dataset: str, test_dataset: str) -> dict:
    y_true = test_df["label"].astype(int).to_numpy()
    y_pred = model.predict(test_df["text"])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result = {
        "model": model_name,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    scores = get_scores(model, test_df["text"])
    if scores is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        result["roc_auc"] = np.nan
    return result
