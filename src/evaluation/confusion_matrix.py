from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(y_true, y_pred, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(cm, display_labels=["Human", "AI"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
