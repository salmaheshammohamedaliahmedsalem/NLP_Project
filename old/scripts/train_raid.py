from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate text-only RAID baselines.")
    parser.add_argument("--data", default="data/raid/raid_balanced_subset.csv")
    parser.add_argument("--results-dir", default="results/raid")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raid_train")
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
        y_true, y_pred, average="macro", zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        result["roc_auc"] = np.nan
    return result


def make_models() -> dict[str, object]:
    return {
        "RAID TF-IDF Word LR": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=80000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ]
        ),
        "RAID TF-IDF Char LR": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=100000,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)),
            ]
        ),
        "RAID TF-IDF Word SVM": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=80000,
                    ),
                ),
                ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=3)),
            ]
        ),
    }


def predict(model, texts):
    y_pred = model.predict(texts)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(texts)[:, 1]
    else:
        y_score = None
    return y_pred, y_score


def random_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label_binary"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_binary"])
    return {"train": train_df, "val": val_df, "test": test_df}


def source_group_split(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label_binary"], groups=df["source_id"].fillna(df["title"])))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42, stratify=train_df["label_binary"])
    return {"train": train_df, "val": val_df, "test": test_df}


def model_holdout_splits(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    splits = {}
    ai_models = [m for m, count in df[df["label_binary"] == 1]["model"].value_counts().items() if count >= 250]
    for model_name in ai_models[:4]:
        test_df = df[(df["model"] == model_name) | ((df["label_binary"] == 0) & (df["domain"].isin(df[df["model"] == model_name]["domain"].unique())))].copy()
        train_df = df[(df["model"] != model_name)].copy()
        if test_df["label_binary"].nunique() < 2 or train_df["label_binary"].nunique() < 2:
            continue
        train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42, stratify=train_df["label_binary"])
        splits[f"holdout_model_{model_name}"] = {"train": train_df, "val": val_df, "test": test_df}
    return splits


def attack_holdout_splits(df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    splits = {}
    attacks = [a for a, count in df[df["label_binary"] == 1]["attack"].value_counts().items() if a != "none" and count >= 250]
    for attack in attacks[:3]:
        test_df = df[(df["attack"] == attack) | (df["label_binary"] == 0)].copy()
        train_df = df[(df["attack"] != attack)].copy()
        if test_df["label_binary"].nunique() < 2 or train_df["label_binary"].nunique() < 2:
            continue
        train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42, stratify=train_df["label_binary"])
        splits[f"holdout_attack_{attack}"] = {"train": train_df, "val": val_df, "test": test_df}
    return splits


def main() -> None:
    args = parse_args()
    data_path = ROOT / args.data
    results_dir = ROOT / args.results_dir
    models_dir = ROOT / args.models_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(results_dir / "raid_training.log")

    logger.info("Loading RAID subset: %s", data_path)
    df = pd.read_csv(data_path).dropna(subset=["text", "label_binary"]).reset_index(drop=True)
    fallback_ids = pd.Series(df.index.astype(str), index=df.index)
    df["source_id"] = df["source_id"].fillna(df["title"]).fillna(fallback_ids)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Label counts: %s", df["label"].value_counts().to_dict())
    logger.info("Domain counts: %s", df["domain"].value_counts().to_dict())
    logger.info("Model counts: %s", df["model"].value_counts().head(15).to_dict())
    logger.info("Attack counts: %s", df["attack"].value_counts().head(15).to_dict())

    splits = {"random": random_split(df), "source_grouped": source_group_split(df)}
    splits.update(model_holdout_splits(df))
    splits.update(attack_holdout_splits(df))

    rows = []
    best_random_model = None
    best_random_score = -1.0
    for split_name, split in splits.items():
        logger.info("=== Split: %s | train=%s val=%s test=%s ===", split_name, len(split["train"]), len(split["val"]), len(split["test"]))
        for model_name, model in make_models().items():
            logger.info("Training %s on %s", model_name, split_name)
            model.fit(split["train"]["text"], split["train"]["label_binary"])
            y_pred, y_score = predict(model, split["test"]["text"])
            result = {
                "split": split_name,
                "model": model_name,
                "train_size": len(split["train"]),
                "test_size": len(split["test"]),
                **metrics(split["test"]["label_binary"], y_pred, y_score),
            }
            rows.append(result)
            logger.info(
                "%s | %s | acc=%.4f f1=%.4f macro_f1=%.4f auc=%.4f",
                split_name,
                model_name,
                result["accuracy"],
                result["f1"],
                result["macro_f1"],
                result["roc_auc"],
            )
            if split_name == "random" and result["macro_f1"] > best_random_score:
                best_random_score = result["macro_f1"]
                best_random_model = model

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "raid_results.csv", index=False)
    if best_random_model is not None:
        joblib.dump(best_random_model, models_dir / "raid_best_tfidf.joblib")
        logger.info("Saved best random-split RAID model to %s", models_dir / "raid_best_tfidf.joblib")
    logger.info("Saved RAID results to %s", results_dir / "raid_results.csv")
    logger.info("\n%s", results.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
