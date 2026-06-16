from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR, RESULTS_DIR
from src.visualization.result_tables import dataframe_to_markdown, save_markdown_table


def maybe_load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, filename, title in [
        ("semeval", "semeval_results.csv", "Table 1: SemEval Results"),
        ("raid", "raid_results.csv", "Table 2: RAID Results"),
        ("cross", "cross_dataset_results.csv", "Table 3: Cross-Dataset Results"),
        ("arabic", "arabic_poc_results.csv", "Table 4: Arabic POC Results"),
    ]:
        df = maybe_load(RESULTS_DIR / filename)
        if df is None:
            continue
        cols = [col for col in ["model", "train_dataset", "test_dataset", "accuracy", "precision", "recall", "f1", "roc_auc"] if col in df.columns]
        table = df[cols].copy()
        save_markdown_table(table, REPORTS_DIR / f"{name}_table.md", title)
        tables.append((title, table))

    lines = ["# Experiment Summary", ""]
    for title, table in tables:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(dataframe_to_markdown(table))
        lines.append("")
    (REPORTS_DIR / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report summary to {REPORTS_DIR / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
