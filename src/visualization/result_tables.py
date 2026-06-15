from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_markdown_table(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"# {title}\n\n" + df.to_markdown(index=False) + "\n"
    output_path.write_text(content, encoding="utf-8")
