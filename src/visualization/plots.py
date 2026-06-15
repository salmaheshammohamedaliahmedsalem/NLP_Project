from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_f1_barplot(results_df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results_df.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(ordered["model"], ordered["f1"], color="#2b6f6c")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_feature_importance(model, feature_names: list[str], output_path: str | Path, title: str) -> None:
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    values.iloc[::-1].plot(kind="barh", ax=ax, color="#5b8c5a")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
