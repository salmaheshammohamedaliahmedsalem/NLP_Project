from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_human_detector.data import clean_dataset, load_raw_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an epoch-based TF-IDF detector with loss logging.")
    parser.add_argument("--data", default="ai_vs_human_content_v2_20000.csv")
    parser.add_argument("--processed", default="data/processed/clean_ai_vs_human_content.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("epoch_training")
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


def load_or_create_clean_data(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    processed_path = ROOT / args.processed
    if processed_path.exists():
        logger.info("Loading cleaned dataset from %s", processed_path)
        return pd.read_csv(processed_path).dropna(subset=["content", "label_binary"]).reset_index(drop=True)

    logger.info("Cleaned dataset missing. Rebuilding from raw dataset.")
    raw_df, text_col, label_col, domain_col = load_raw_dataset(ROOT / args.data)
    clean_df, audit = clean_dataset(raw_df, text_col, label_col, domain_col)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)
    logger.info("Created cleaned dataset with %s rows. Audit: %s", len(clean_df), audit)
    return clean_df


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def write_loss_curve_svg(history: pd.DataFrame, output_path: Path) -> None:
    width = 900
    height = 420
    padding_left = 72
    padding_right = 28
    padding_top = 34
    padding_bottom = 58
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    epochs = history["epoch"].astype(float).to_numpy()
    train_loss = history["train_loss"].astype(float).to_numpy()
    val_loss = history["val_loss"].astype(float).to_numpy()
    min_x, max_x = float(epochs.min()), float(epochs.max())
    min_y = float(min(train_loss.min(), val_loss.min()))
    max_y = float(max(train_loss.max(), val_loss.max()))
    y_span = max(max_y - min_y, 1e-9)
    min_y = max(0.0, min_y - y_span * 0.08)
    max_y = max_y + y_span * 0.08

    def x_coord(value: float) -> float:
        return padding_left + (value - min_x) / max(max_x - min_x, 1e-9) * plot_width

    def y_coord(value: float) -> float:
        return padding_top + (max_y - value) / max(max_y - min_y, 1e-9) * plot_height

    def polyline(values: np.ndarray) -> str:
        return " ".join(f"{x_coord(epoch):.2f},{y_coord(value):.2f}" for epoch, value in zip(epochs, values))

    grid_lines = []
    for index in range(6):
        ratio = index / 5
        y_value = min_y + (max_y - min_y) * ratio
        y = y_coord(y_value)
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.2f}" x2="{width - padding_right}" y2="{y:.2f}" stroke="#e1e1dc" />'
        )
        grid_lines.append(
            f'<text x="{padding_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#555">{y_value:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f4"/>
<text x="{padding_left}" y="22" font-size="18" font-family="Arial" font-weight="700" fill="#1d1d1b">Training and Validation Loss over Epochs</text>
{''.join(grid_lines)}
<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}" stroke="#777"/>
<line x1="{padding_left}" y1="{height - padding_bottom}" x2="{width - padding_right}" y2="{height - padding_bottom}" stroke="#777"/>
<polyline points="{polyline(train_loss)}" fill="none" stroke="#2b6f6c" stroke-width="3"/>
<polyline points="{polyline(val_loss)}" fill="none" stroke="#b24c3f" stroke-width="3"/>
<text x="{padding_left}" y="{height - 22}" font-size="13" font-family="Arial" fill="#555">Epoch</text>
<text x="18" y="{padding_top + 12}" font-size="13" font-family="Arial" fill="#555" transform="rotate(-90 18,{padding_top + 12})">Loss</text>
<rect x="{width - 244}" y="42" width="206" height="58" rx="8" fill="#ffffff" stroke="#d9d8d0"/>
<line x1="{width - 226}" y1="64" x2="{width - 184}" y2="64" stroke="#2b6f6c" stroke-width="3"/>
<text x="{width - 174}" y="68" font-size="13" font-family="Arial" fill="#1d1d1b">Training loss</text>
<line x1="{width - 226}" y1="86" x2="{width - 184}" y2="86" stroke="#b24c3f" stroke-width="3"/>
<text x="{width - 174}" y="90" font-size="13" font-family="Arial" fill="#1d1d1b">Validation loss</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = ROOT / args.results_dir
    models_dir = ROOT / args.models_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(results_dir / "training.log")

    logger.info("Starting epoch-based detector training")
    logger.info("Epochs: %s", args.epochs)

    df = load_or_create_clean_data(args, logger)
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label_binary"],
    )
    logger.info("Train rows: %s | Validation rows: %s", len(train_df), len(val_df))
    logger.info("Train label distribution: %s", train_df["label_binary"].value_counts().sort_index().to_dict())
    logger.info("Validation label distribution: %s", val_df["label_binary"].value_counts().sort_index().to_dict())

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )
    x_train = vectorizer.fit_transform(train_df["content"])
    x_val = vectorizer.transform(val_df["content"])
    y_train = train_df["label_binary"].astype(int).to_numpy()
    y_val = val_df["label_binary"].astype(int).to_numpy()

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = {label: len(y_train) / (2 * count) for label, count in enumerate(class_counts) if count}
    sample_weights = np.array([class_weights[label] for label in y_train])

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        learning_rate="optimal",
        random_state=42,
        fit_intercept=True,
    )

    rng = np.random.default_rng(42)
    rows = []
    classes = np.array([0, 1])
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y_train))
        classifier.partial_fit(
            x_train[order],
            y_train[order],
            classes=classes,
            sample_weight=sample_weights[order],
        )

        train_prob = classifier.predict_proba(x_train)[:, 1]
        val_prob = classifier.predict_proba(x_val)[:, 1]
        train_loss = log_loss(y_train, train_prob, labels=classes)
        val_loss = log_loss(y_val, val_prob, labels=classes)
        train_metrics = compute_metrics(y_train, train_prob)
        val_metrics = compute_metrics(y_val, val_prob)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        rows.append(row)
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | train_f1=%.4f | val_f1=%.4f | val_auc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            train_metrics["f1"],
            val_metrics["f1"],
            val_metrics["roc_auc"],
        )

    history = pd.DataFrame(rows)
    history_path = results_dir / "training_history.csv"
    history.to_csv(history_path, index=False)
    loss_curve_path = results_dir / "training_loss_curve.svg"
    write_loss_curve_svg(history, loss_curve_path)
    model_path = models_dir / "epoch_tfidf_sgd.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, model_path)

    best = history.sort_values(["val_f1", "val_roc_auc"], ascending=False).iloc[0]
    logger.info("Saved training history to %s", history_path)
    logger.info("Saved loss curve diagram to %s", loss_curve_path)
    logger.info("Saved model artifact to %s", model_path)
    logger.info(
        "Best epoch=%s | val_loss=%.4f | val_f1=%.4f | val_auc=%.4f",
        int(best["epoch"]),
        best["val_loss"],
        best["val_f1"],
        best["val_roc_auc"],
    )


if __name__ == "__main__":
    main()
