from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from src.evaluation.confusion_matrix import save_confusion_matrix  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.models.save_load import save_model  # noqa: E402
from src.models.train_models import train_model  # noqa: E402
from src.visualization.plots import save_f1_barplot  # noqa: E402


ARABIC_MODELS = ("M2_Char_TFIDF_LinearSVM", "M4_Hybrid_TFIDF_Stylometric")


def main() -> None:
    split_dir = SPLITS_DIR / "arabic"
    if not (split_dir / "train.csv").exists():
        print("Arabic split not found. Run preprocessing with an Arabic dataset first.")
        return
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    rows = []
    for model_name in ARABIC_MODELS:
        print(f"Training Arabic POC model {model_name}")
        model = train_model(model_name, train_df)
        result = evaluate_model(model, test_df, model_name, "arabic", "arabic")
        rows.append(result)
        predictions = model.predict(test_df["text"])
        save_model(model, MODELS_DIR / "arabic" / f"{model_name}.joblib")
        save_confusion_matrix(
            test_df["label"],
            predictions,
            FIGURES_DIR / "confusion_matrices" / "arabic" / f"{model_name}.png",
            f"Arabic POC: {model_name}",
        )
        print(result)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "arabic_poc_results.csv", index=False)
    save_f1_barplot(results, FIGURES_DIR / "comparison_plots" / "arabic_poc_f1.png", "Arabic POC F1 Comparison")


if __name__ == "__main__":
    main()
