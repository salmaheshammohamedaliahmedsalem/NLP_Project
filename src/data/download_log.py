from __future__ import annotations

from pathlib import Path

from src.config import DOWNLOAD_LOG_PATH


def init_download_log(path: str | Path = DOWNLOAD_LOG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Dataset Download Log\n\n", encoding="utf-8")


def append_download_log(
    dataset_name: str,
    source: str,
    command: str,
    success: bool,
    local_raw_path: str | Path,
    notes: str = "",
    path: str | Path = DOWNLOAD_LOG_PATH,
) -> None:
    init_download_log(path)
    status = "succeeded" if success else "failed"
    entry = (
        f"## {dataset_name}\n\n"
        f"- Source: {source}\n"
        f"- Command: `{command}`\n"
        f"- Status: {status}\n"
        f"- Local raw path: `{local_raw_path}`\n"
        f"- Notes/errors: {notes or 'None'}\n\n"
    )
    Path(path).write_text(Path(path).read_text(encoding="utf-8") + entry, encoding="utf-8")
