from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def format_label_distribution(df: pd.DataFrame) -> str:
    counts = df["label_binary"].value_counts(normalize=True).sort_index()
    return "; ".join(f"{int(label)}={share:.3f}" for label, share in counts.items())
