from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_human_detector.data import (  # noqa: E402
    clean_dataset,
    load_raw_dataset,
    make_domain_holdout_splits,
    make_prompt_grouped_split,
    make_random_split,
    metadata_leakage_report,
)
from ai_human_detector.evaluation import binary_metrics, format_label_distribution  # noqa: E402
from ai_human_detector.features import extract_stylometric_frame  # noqa: E402
from ai_human_detector.models import model_registry, predict_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-vs-human detection experiments.")
    parser.add_argument("--data", default="ai_vs_human_content_v2_20000.csv")
    parser.add_argument("--challenge", default="challenge_test.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap per split for quick debugging.")
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def near_duplicate_audit(df: pd.DataFrame, output_path: Path, sample_size: int = 1200) -> pd.DataFrame:
    sample = df.sample(min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    if len(sample) < 2:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=1, max_features=20000)
    matrix = vectorizer.fit_transform(sample["content"])
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    pairs = np.argwhere(similarity >= 0.90)
    rows = []
    seen = set()
    for left, right in pairs[:50]:
        key = tuple(sorted((int(left), int(right))))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "left_index": int(sample.loc[left].name),
                "right_index": int(sample.loc[right].name),
                "similarity": float(similarity[left, right]),
                "left_label": int(sample.loc[left, "label_binary"]),
                "right_label": int(sample.loc[right, "label_binary"]),
                "left_preview": sample.loc[left, "content"][:180],
                "right_preview": sample.loc[right, "content"][:180],
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    return report


def evaluate_model(model_name: str, model, split_name: str, split: dict[str, pd.DataFrame]) -> dict[str, object]:
    start = time.time()
    train_df = split["train"]
    test_df = split["test"]
    model.fit(train_df["content"], train_df["label_binary"])
    predictions, scores = predict_scores(model, test_df["content"])
    metrics = binary_metrics(test_df["label_binary"], predictions, scores)
    return {
        "split": split_name,
        "model": model_name,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_label_distribution": format_label_distribution(train_df),
        "test_label_distribution": format_label_distribution(test_df),
        "seconds": round(time.time() - start, 2),
        **metrics,
    }


def run_sanity_checks(clean_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    train_df, test_df = train_test_split(
        clean_df,
        test_size=0.20,
        random_state=123,
        stratify=clean_df["label_binary"],
    )

    checks = []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    x_train = vectorizer.fit_transform(train_df["content"])
    x_test = vectorizer.transform(test_df["content"])
    shuffled_labels = np.random.default_rng(42).permutation(train_df["label_binary"].values)
    shuffled_model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    shuffled_model.fit(x_train, shuffled_labels)
    shuffled_pred = shuffled_model.predict(x_test)
    checks.append({"check": "label_shuffle_text_model", **binary_metrics(test_df["label_binary"], shuffled_pred)})

    train_sty = extract_stylometric_frame(train_df["content"])
    test_sty = extract_stylometric_frame(test_df["content"])
    length_cols = ["char_count", "word_count", "sentence_count", "line_count"]
    scaler = StandardScaler()
    length_model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    length_model.fit(scaler.fit_transform(train_sty[length_cols]), train_df["label_binary"])
    length_pred = length_model.predict(scaler.transform(test_sty[length_cols]))
    length_prob = length_model.predict_proba(scaler.transform(test_sty[length_cols]))[:, 1]
    checks.append({"check": "length_only_model", **binary_metrics(test_df["label_binary"], length_pred, length_prob)})

    if "prompt_group" in clean_df.columns:
        prompt_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        x_prompt_train = prompt_vectorizer.fit_transform(train_df["prompt_group"])
        x_prompt_test = prompt_vectorizer.transform(test_df["prompt_group"])
        prompt_model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
        prompt_model.fit(x_prompt_train, train_df["label_binary"])
        prompt_pred = prompt_model.predict(x_prompt_test)
        prompt_prob = prompt_model.predict_proba(x_prompt_test)[:, 1]
        checks.append({"check": "prompt_only_model", **binary_metrics(test_df["label_binary"], prompt_pred, prompt_prob)})

    report = pd.DataFrame(checks)
    report.to_csv(output_path, index=False)
    return report


def evaluate_challenge(challenge_path: Path, trained_models: dict[str, object], output_path: Path) -> pd.DataFrame:
    if not challenge_path.exists():
        return pd.DataFrame()
    challenge = pd.read_csv(challenge_path).dropna(subset=["content", "label"]).copy()
    challenge["label_binary"] = challenge["label"].apply(lambda value: 1 if str(value).strip().lower() == "ai" else 0)
    rows = []
    for name, model in trained_models.items():
        predictions, scores = predict_scores(model, challenge["content"])
        rows.append({"split": "challenge", "model": name, "test_size": len(challenge), **binary_metrics(challenge["label_binary"], predictions, scores)})
    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False)
    return report


def main() -> None:
    args = parse_args()
    data_path = ROOT / args.data
    challenge_path = ROOT / args.challenge
    results_dir = ROOT / args.results_dir
    processed_dir = ROOT / args.processed_dir
    ensure_dirs(results_dir, processed_dir)

    raw_df, text_col, label_col, domain_col = load_raw_dataset(data_path)
    leakage = metadata_leakage_report(raw_df, label_col, results_dir / "metadata_leakage_report.csv")
    clean_df, audit = clean_dataset(raw_df, text_col, label_col, domain_col)
    clean_df.to_csv(processed_dir / "clean_ai_vs_human_content.csv", index=False)
    (results_dir / "cleaning_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    near_duplicate_audit(clean_df, results_dir / "near_duplicate_audit.csv")

    splits: dict[str, dict[str, pd.DataFrame]] = {"random": make_random_split(clean_df)}
    prompt_split = make_prompt_grouped_split(clean_df)
    if prompt_split is not None:
        splits["prompt_grouped"] = prompt_split
    splits.update(make_domain_holdout_splits(clean_df))

    rows = []
    trained_random_models = {}
    for split_name, split in splits.items():
        if args.max_train and len(split["train"]) > args.max_train:
            split = {
                key: value.sample(min(len(value), args.max_train if key == "train" else len(value)), random_state=42).reset_index(drop=True)
                for key, value in split.items()
            }
        for model_name, model in model_registry().items():
            print(f"Training {model_name} on {split_name}...")
            result = evaluate_model(model_name, model, split_name, split)
            rows.append(result)
            print(f"  F1={result['f1']:.4f}, accuracy={result['accuracy']:.4f}, seconds={result['seconds']}")
            if split_name == "random":
                trained_random_models[model_name] = model

    experiment_results = pd.DataFrame(rows)
    experiment_results.to_csv(results_dir / "experiment_results.csv", index=False)
    run_sanity_checks(clean_df, results_dir / "sanity_checks.csv")
    challenge_results = evaluate_challenge(challenge_path, trained_random_models, results_dir / "challenge_test_results_new.csv")

    if not challenge_results.empty:
        combined = pd.concat([experiment_results, challenge_results], ignore_index=True, sort=False)
    else:
        combined = experiment_results
    combined.to_csv(results_dir / "all_results_research_pipeline.csv", index=False)

    print("\n=== Dataset audit ===")
    print(json.dumps(audit, indent=2))
    print("\n=== Leakage proxy report ===")
    print(leakage.head(10).to_string(index=False))
    print("\n=== Results ===")
    print(combined.sort_values(["split", "f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
