from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced, manageable RAID subset from Hugging Face streaming.")
    parser.add_argument("--dataset", default="liamdugan/raid")
    parser.add_argument("--config", default="raid")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/raid/raid_balanced_subset.csv")
    parser.add_argument("--max-human", type=int, default=5000)
    parser.add_argument("--max-ai", type=int, default=5000)
    parser.add_argument("--max-per-domain-label", type=int, default=900)
    parser.add_argument("--max-per-model", type=int, default=800)
    parser.add_argument("--max-scan", type=int, default=350000)
    return parser.parse_args()


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("raid_prepare")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").split())


def should_keep(row: dict, counters: dict, args: argparse.Namespace) -> bool:
    model = str(row.get("model", "") or "")
    label = "human" if model.lower() == "human" else "ai"
    domain = str(row.get("domain", "unknown") or "unknown")
    text = normalize_text(row.get("generation", ""))
    if len(text.split()) < 30:
        return False
    if label == "human" and counters["label"][label] >= args.max_human:
        return False
    if label == "ai" and counters["label"][label] >= args.max_ai:
        return False
    if counters["domain_label"][(domain, label)] >= args.max_per_domain_label:
        return False
    if label == "ai" and counters["model"][model] >= args.max_per_model:
        return False
    return True


def main() -> None:
    args = parse_args()
    logger = configure_logger()
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RAID with streaming: dataset=%s config=%s split=%s", args.dataset, args.config, args.split)
    logger.info(
        "Targets: max_human=%s max_ai=%s max_per_domain_label=%s max_per_model=%s max_scan=%s",
        args.max_human,
        args.max_ai,
        args.max_per_domain_label,
        args.max_per_model,
        args.max_scan,
    )

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    rows = []
    counters = {
        "label": Counter(),
        "domain_label": Counter(),
        "model": Counter(),
        "attack": Counter(),
        "domain": Counter(),
    }
    seen_texts = set()

    for index, row in enumerate(dataset, start=1):
        if index > args.max_scan:
            logger.info("Reached max_scan=%s", args.max_scan)
            break

        text = normalize_text(row.get("generation", ""))
        if not text or text in seen_texts:
            continue
        if not should_keep(row, counters, args):
            continue

        model = str(row.get("model", "") or "")
        label = "human" if model.lower() == "human" else "ai"
        domain = str(row.get("domain", "unknown") or "unknown")
        attack = str(row.get("attack", "none") or "none")
        decoding = str(row.get("decoding", "unknown") or "unknown")

        seen_texts.add(text)
        rows.append(
            {
                "text": text,
                "label": label,
                "label_binary": 0 if label == "human" else 1,
                "model": model,
                "domain": domain,
                "attack": attack,
                "decoding": decoding,
                "repetition_penalty": str(row.get("repetition_penalty", "unknown") or "unknown"),
                "source_id": row.get("source_id"),
                "title": row.get("title"),
            }
        )
        counters["label"][label] += 1
        counters["domain_label"][(domain, label)] += 1
        counters["model"][model] += 1
        counters["attack"][attack] += 1
        counters["domain"][domain] += 1

        if len(rows) % 1000 == 0:
            logger.info(
                "Kept %s rows after scanning %s | labels=%s | domains=%s",
                len(rows),
                index,
                dict(counters["label"]),
                dict(counters["domain"]),
            )
        if counters["label"]["human"] >= args.max_human and counters["label"]["ai"] >= args.max_ai:
            logger.info("Reached label targets after scanning %s rows", index)
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No RAID rows were collected. Check network access or dataset schema.")

    df.to_csv(output_path, index=False)
    logger.info("Saved RAID subset to %s", output_path)
    logger.info("Final shape: %s", df.shape)
    logger.info("Label counts: %s", df["label"].value_counts().to_dict())
    logger.info("Domain counts: %s", df["domain"].value_counts().to_dict())
    logger.info("Model counts: %s", df["model"].value_counts().head(20).to_dict())
    logger.info("Attack counts: %s", df["attack"].value_counts().head(20).to_dict())


if __name__ == "__main__":
    main()
