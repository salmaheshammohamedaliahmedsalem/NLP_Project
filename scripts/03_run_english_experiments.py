from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, MODEL_NAMES, MODELS_DIR, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from src.evaluation.confusion_matrix import save_confusion_matrix  # noqa: E402
from src.evaluation.error_analysis import save_errors  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.features.stylometric_features import extract_stylometric_features  # noqa: E402
from src.models.save_load import save_model  # noqa: E402
from src.models.train_models import train_model  # noqa: E402
from src.visualization.plots import save_f1_barplot, save_feature_importance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run within-dataset English experiments.")
    parser.add_argument("--dataset", choices=["semeval", "raid"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = SPLITS_DIR / args.dataset
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    rows = []
    for model_name in MODEL_NAMES:
        print(f"Training {model_name} on {args.dataset}")
        model = train_model(model_name, train_df)
        result = evaluate_model(model, test_df, model_name, args.dataset, args.dataset)
        rows.append(result)
        predictions = model.predict(test_df["text"])
        save_model(model, MODELS_DIR / args.dataset / f"{model_name}.joblib")
        save_confusion_matrix(
            test_df["label"],
            predictions,
            FIGURES_DIR / "confusion_matrices" / args.dataset / f"{model_name}.png",
            f"{args.dataset}: {model_name}",
        )
        save_errors(model, test_df, RESULTS_DIR / f"errors_{args.dataset}_{model_name}.csv")
        if model_name == "M3_Stylometric_RandomForest":
            feature_names = list(extract_stylometric_features(train_df["text"]).columns)
            save_feature_importance(
                model,
                feature_names,
                FIGURES_DIR / "feature_importance" / f"{args.dataset}_{model_name}.png",
                f"{args.dataset} stylometric feature importance",
            )
        print(result)
    results = pd.DataFrame(rows)
    output_name = "semeval_results.csv" if args.dataset == "semeval" else "raid_results.csv"
    results.to_csv(RESULTS_DIR / output_name, index=False)
    save_f1_barplot(results, FIGURES_DIR / "comparison_plots" / f"{args.dataset}_f1.png", f"{args.dataset.upper()} F1 Comparison")
    print(f"Saved results to {RESULTS_DIR / output_name}")


if __name__ == "__main__":
    main()
