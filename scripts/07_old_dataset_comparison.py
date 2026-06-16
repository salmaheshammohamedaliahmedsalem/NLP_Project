from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, MODEL_NAMES, MODELS_DIR, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from src.evaluation.confusion_matrix import save_confusion_matrix  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.models.save_load import load_model, save_model  # noqa: E402
from src.models.train_models import train_model  # noqa: E402
from src.visualization.plots import save_f1_barplot  # noqa: E402


def load_split(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / dataset / f"{split}.csv")


def evaluate_saved_models_on_old() -> list[dict]:
    old_test = load_split("old_ai", "test")
    rows: list[dict] = []
    for train_dataset in ("semeval", "raid"):
        for model_name in MODEL_NAMES:
            model_path = MODELS_DIR / train_dataset / f"{model_name}.joblib"
            if not model_path.exists():
                print(f"Skipping missing saved model: {model_path}")
                continue
            model = load_model(model_path)
            result = evaluate_model(model, old_test, model_name, train_dataset, "old_ai")
            rows.append(result)
            predictions = model.predict(old_test["text"])
            save_confusion_matrix(
                old_test["label"],
                predictions,
                FIGURES_DIR / "confusion_matrices" / f"cross_{train_dataset}_to_old_ai_{model_name}.png",
                f"{train_dataset} to old_ai: {model_name}",
            )
            print(result)
    return rows


def train_old_and_evaluate_new_tests() -> list[dict]:
    old_train = load_split("old_ai", "train")
    rows: list[dict] = []
    for model_name in MODEL_NAMES:
        print(f"Training {model_name} on old_ai")
        model = train_model(model_name, old_train)
        save_model(model, MODELS_DIR / "old_ai" / f"{model_name}.joblib")
        for test_dataset in ("semeval", "raid"):
            test_df = load_split(test_dataset, "test")
            result = evaluate_model(model, test_df, model_name, "old_ai", test_dataset)
            rows.append(result)
            predictions = model.predict(test_df["text"])
            save_confusion_matrix(
                test_df["label"],
                predictions,
                FIGURES_DIR / "confusion_matrices" / f"cross_old_ai_to_{test_dataset}_{model_name}.png",
                f"old_ai to {test_dataset}: {model_name}",
            )
            print(result)
    return rows


def main() -> None:
    rows = []
    rows.extend(evaluate_saved_models_on_old())
    rows.extend(train_old_and_evaluate_new_tests())
    results = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "old_cross_dataset_results.csv"
    results.to_csv(output_path, index=False)
    save_f1_barplot(results, FIGURES_DIR / "comparison_plots" / "old_cross_dataset_f1.png", "Old Dataset Cross-Evaluation F1")
    print(f"Saved old cross-dataset results to {output_path}")


if __name__ == "__main__":
    main()
