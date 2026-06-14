from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_human_detector.models import model_registry, predict_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate previous project models on a RAID benchmark subset.")
    parser.add_argument("--train-data", default="data/processed/clean_ai_vs_human_content.csv")
    parser.add_argument("--raid-data", default="data/raid/raid_quick_subset.csv")
    parser.add_argument("--results-dir", default="results/raid")
    parser.add_argument("--saved-epoch-model", default="models/epoch_tfidf_sgd.joblib")
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("previous_on_raid")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        output["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        output["roc_auc"] = np.nan
    return output


def predict_saved_epoch_model(model_bundle: dict, texts: pd.Series):
    matrix = model_bundle["vectorizer"].transform(texts)
    classifier = model_bundle["classifier"]
    predictions = classifier.predict(matrix)
    scores = classifier.predict_proba(matrix)[:, 1]
    return predictions, scores


def slice_rows(raid_df: pd.DataFrame, model_name: str, y_pred, y_score) -> list[dict[str, object]]:
    rows = []
    eval_df = raid_df.copy()
    eval_df["_pred"] = y_pred
    eval_df["_score"] = y_score if y_score is not None else np.nan

    for column in ["domain", "model", "attack", "decoding"]:
        if column not in eval_df.columns:
            continue
        for value, group in eval_df.groupby(column):
            if len(group) < 20 or group["label_binary"].nunique() < 2:
                continue
            group_metrics = metrics(group["label_binary"], group["_pred"], group["_score"])
            rows.append(
                {
                    "detector": model_name,
                    "slice_column": column,
                    "slice_value": value,
                    "n": len(group),
                    **group_metrics,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    results_dir = ROOT / args.results_dir
    logger = configure_logger(results_dir / "previous_models_on_raid.log")

    train_path = ROOT / args.train_data
    raid_path = ROOT / args.raid_data
    logger.info("Loading previous cleaned training data: %s", train_path)
    train_df = pd.read_csv(train_path).dropna(subset=["content", "label_binary"]).reset_index(drop=True)
    logger.info("Loading RAID benchmark subset: %s", raid_path)
    raid_df = pd.read_csv(raid_path).dropna(subset=["text", "label_binary"]).reset_index(drop=True)

    logger.info("Previous training shape: %s", train_df.shape)
    logger.info("Previous training label counts: %s", train_df["label_binary"].value_counts().sort_index().to_dict())
    logger.info("RAID benchmark shape: %s", raid_df.shape)
    logger.info("RAID label counts: %s", raid_df["label_binary"].value_counts().sort_index().to_dict())
    logger.info("RAID domains: %s", raid_df["domain"].value_counts().to_dict())
    logger.info("RAID generators: %s", raid_df["model"].value_counts().to_dict())

    rows = []
    slice_output = []
    for model_name, model in model_registry().items():
        logger.info("Training previous model on original cleaned data: %s", model_name)
        model.fit(train_df["content"], train_df["label_binary"])
        y_pred, y_score = predict_scores(model, raid_df["text"])
        result = {
            "train_dataset": "original_cleaned_global_ai_vs_human",
            "test_dataset": "raid_quick_subset",
            "detector": model_name,
            "train_size": len(train_df),
            "test_size": len(raid_df),
            **metrics(raid_df["label_binary"], y_pred, y_score),
        }
        rows.append(result)
        slice_output.extend(slice_rows(raid_df, model_name, y_pred, y_score))
        logger.info(
            "%s on RAID | acc=%.4f f1=%.4f macro_f1=%.4f auc=%.4f",
            model_name,
            result["accuracy"],
            result["f1"],
            result["macro_f1"],
            result["roc_auc"],
        )

    saved_path = ROOT / args.saved_epoch_model
    if saved_path.exists():
        logger.info("Evaluating saved epoch-trained app model: %s", saved_path)
        bundle = joblib.load(saved_path)
        y_pred, y_score = predict_saved_epoch_model(bundle, raid_df["text"])
        result = {
            "train_dataset": "original_cleaned_global_ai_vs_human",
            "test_dataset": "raid_quick_subset",
            "detector": "Saved Epoch TF-IDF SGD",
            "train_size": len(train_df),
            "test_size": len(raid_df),
            **metrics(raid_df["label_binary"], y_pred, y_score),
        }
        rows.append(result)
        slice_output.extend(slice_rows(raid_df, "Saved Epoch TF-IDF SGD", y_pred, y_score))
        logger.info(
            "Saved Epoch TF-IDF SGD on RAID | acc=%.4f f1=%.4f macro_f1=%.4f auc=%.4f",
            result["accuracy"],
            result["f1"],
            result["macro_f1"],
            result["roc_auc"],
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    slices = pd.DataFrame(slice_output)
    results.to_csv(results_dir / "previous_models_on_raid.csv", index=False)
    slices.to_csv(results_dir / "previous_models_on_raid_slices.csv", index=False)

    logger.info("Saved summary to %s", results_dir / "previous_models_on_raid.csv")
    logger.info("Saved slices to %s", results_dir / "previous_models_on_raid_slices.csv")
    logger.info("\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
