from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_human_detector.data import make_prompt_grouped_split, make_random_split  # noqa: E402
from ai_human_detector.evaluation import binary_metrics  # noqa: E402
from ai_human_detector.features import extract_stylometric_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen transformer embedding experiments.")
    parser.add_argument("--data", default="data/processed/clean_ai_vs_human_content.csv")
    parser.add_argument("--challenge", default="challenge_test.csv")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-train", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size: int, max_length: int) -> np.ndarray:
    vectors = []
    model.eval()
    text_list = list(texts)
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def fit_eval_embedding_lr(train_df, test_df, tokenizer, encoder, device, args, hybrid: bool = False):
    train_embeddings = encode_texts(train_df["content"], tokenizer, encoder, device, args.batch_size, args.max_length)
    test_embeddings = encode_texts(test_df["content"], tokenizer, encoder, device, args.batch_size, args.max_length)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_embeddings)
    x_test = scaler.transform(test_embeddings)

    if hybrid:
        sty_scaler = StandardScaler()
        train_sty = sty_scaler.fit_transform(extract_stylometric_frame(train_df["content"]))
        test_sty = sty_scaler.transform(extract_stylometric_frame(test_df["content"]))
        x_train = hstack([csr_matrix(x_train), csr_matrix(train_sty)]).tocsr()
        x_test = hstack([csr_matrix(x_test), csr_matrix(test_sty)]).tocsr()

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    classifier.fit(x_train, train_df["label_binary"])
    predictions = classifier.predict(x_test)
    scores = classifier.predict_proba(x_test)[:, 1]
    return classifier, scaler, binary_metrics(test_df["label_binary"], predictions, scores)


def main() -> None:
    args = parse_args()
    clean_df = pd.read_csv(ROOT / args.data)
    splits = {"random": make_random_split(clean_df)}
    prompt_split = make_prompt_grouped_split(clean_df)
    if prompt_split is not None:
        splits["prompt_grouped"] = prompt_split

    if args.max_train:
        for split in splits.values():
            split["train"] = split["train"].sample(min(args.max_train, len(split["train"])), random_state=42).reset_index(drop=True)

    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)

    rows = []
    random_embedding_model = None
    random_hybrid_model = None
    random_scaler = None

    for split_name, split in splits.items():
        for hybrid in [False, True]:
            model_name = "Frozen DistilRoBERTa Embedding LR"
            if hybrid:
                model_name = "Frozen DistilRoBERTa + Stylometric LR"
            print(f"Training {model_name} on {split_name} with {device}...")
            start = time.time()
            classifier, scaler, metrics = fit_eval_embedding_lr(
                split["train"], split["test"], tokenizer, encoder, device, args, hybrid=hybrid
            )
            rows.append(
                {
                    "split": split_name,
                    "model": model_name,
                    "train_size": len(split["train"]),
                    "test_size": len(split["test"]),
                    "seconds": round(time.time() - start, 2),
                    **metrics,
                }
            )
            print(f"  F1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
            if split_name == "random" and not hybrid:
                random_embedding_model = classifier
                random_scaler = scaler
            if split_name == "random" and hybrid:
                random_hybrid_model = classifier

    challenge_path = ROOT / args.challenge
    if challenge_path.exists() and random_embedding_model is not None and random_scaler is not None:
        challenge = pd.read_csv(challenge_path).dropna(subset=["content", "label"]).copy()
        challenge["label_binary"] = challenge["label"].apply(lambda value: 1 if str(value).strip().lower() == "ai" else 0)
        challenge_embeddings = encode_texts(
            challenge["content"], tokenizer, encoder, device, args.batch_size, args.max_length
        )
        challenge_x = random_scaler.transform(challenge_embeddings)
        predictions = random_embedding_model.predict(challenge_x)
        scores = random_embedding_model.predict_proba(challenge_x)[:, 1]
        rows.append(
            {
                "split": "challenge",
                "model": "Frozen DistilRoBERTa Embedding LR",
                "train_size": np.nan,
                "test_size": len(challenge),
                "seconds": np.nan,
                **binary_metrics(challenge["label_binary"], predictions, scores),
            }
        )

    report = pd.DataFrame(rows)
    output = ROOT / "results" / "transformer_embedding_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
