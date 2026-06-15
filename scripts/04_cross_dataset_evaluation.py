from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, MODEL_NAMES, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from src.evaluation.confusion_matrix import save_confusion_matrix  # noqa: E402
from src.evaluation.error_analysis import save_errors  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.models.train_models import train_model  # noqa: E402
from src.visualization.plots import save_f1_barplot  # noqa: E402


def run_direction(train_name: str, test_name: str) -> list[dict]:
    train_df = pd.read_csv(SPLITS_DIR / train_name / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / test_name / "test.csv")
    rows = []
    for model_name in MODEL_NAMES:
        print(f"Training {model_name} on {train_name}, testing on {test_name}")
        model = train_model(model_name, train_df)
        result = evaluate_model(model, test_df, model_name, train_name, test_name)
        rows.append(result)
        predictions = model.predict(test_df["text"])
        save_confusion_matrix(
            test_df["label"],
            predictions,
            FIGURES_DIR / "confusion_matrices" / f"cross_{train_name}_to_{test_name}_{model_name}.png",
            f"{train_name} to {test_name}: {model_name}",
        )
        save_errors(model, test_df, RESULTS_DIR / f"errors_cross_{train_name}_to_{test_name}_{model_name}.csv")
        print(result)
    return rows


def main() -> None:
    rows = []
    rows.extend(run_direction("semeval", "raid"))
    rows.extend(run_direction("raid", "semeval"))
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "cross_dataset_results.csv", index=False)
    save_f1_barplot(results, FIGURES_DIR / "comparison_plots" / "cross_dataset_f1.png", "Cross-Dataset F1 Comparison")
    print(f"Saved cross-dataset results to {RESULTS_DIR / 'cross_dataset_results.csv'}")


if __name__ == "__main__":
    main()
