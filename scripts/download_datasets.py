from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download_log import append_download_log, init_download_log  # noqa: E402


SEMEVAL_FOLDER_ID = "1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc"
ARABIC_ABSTRACTS = "KFUPM-JRCAI/arabic-generated-abstracts"
ARABIC_SOCIAL = "KFUPM-JRCAI/arabic-generated-social-media-posts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw datasets into data/raw without overwriting preprocessing outputs.")
    parser.add_argument("--skip-semeval", action="store_true")
    parser.add_argument("--skip-raid", action="store_true")
    parser.add_argument("--skip-arabic", action="store_true")
    parser.add_argument("--include-arabic-social", action="store_true")
    parser.add_argument("--raid-max-scan", type=int, default=100_000)
    return parser.parse_args()


def run_command(command: list[str], timeout_seconds: int = 120) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
        return True, (completed.stdout + "\n" + completed.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = stdout + "\n" + stderr
        return False, f"Timed out after {timeout_seconds} seconds.\n{output}".strip()
    except Exception as exc:
        output = ""
        if hasattr(exc, "stdout"):
            output += str(exc.stdout or "")
        if hasattr(exc, "stderr"):
            output += "\n" + str(exc.stderr or "")
        return False, (output.strip() or repr(exc))


def download_semeval() -> None:
    raw_dir = ROOT / "data" / "raw" / "semeval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown", "--folder", f"https://drive.google.com/drive/folders/{SEMEVAL_FOLDER_ID}", "-O", str(raw_dir)]
    print("Downloading SemEval Subtask A with gdown...", flush=True)
    success, notes = run_command(command)
    if not success:
        notes += (
            "\nManual fallback: download SemEval-2024 Task 8 Subtask A from the official source "
            "and place files under data/raw/semeval/."
        )
    append_download_log(
        dataset_name="SemEval-2024 Task 8 Subtask A",
        source=f"Google Drive folder {SEMEVAL_FOLDER_ID}",
        command=" ".join(command),
        success=success,
        local_raw_path=raw_dir,
        notes=notes,
    )
    print("SemEval download", "succeeded" if success else "failed; see outputs/reports/dataset_download_log.md", flush=True)


def load_raid_stream(max_scan: int) -> pd.DataFrame:
    from datasets import load_dataset

    try:
        dataset = load_dataset("liamdugan/raid", "raid", split="train", streaming=True)
    except Exception:
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(dataset, start=1):
        rows.append(row)
        if idx % 25_000 == 0:
            print(f"Downloaded {idx} streamed RAID rows...", flush=True)
        if idx >= max_scan:
            break
    return pd.DataFrame(rows)


def download_raid(max_scan: int) -> None:
    raw_dir = ROOT / "data" / "raw" / "raid"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = f"datasets.load_dataset('liamdugan/raid', split='train', streaming=True), first {max_scan} rows"
    try:
        print("Streaming RAID train rows from Hugging Face...", flush=True)
        print("Important: use RAID train data for supervised experiments because benchmark test labels may be hidden.", flush=True)
        train_df = load_raid_stream(max_scan=max_scan)
        out_path = raw_dir / f"raid_train_stream_{max_scan}.parquet"
        train_df.to_parquet(out_path, index=False)
        append_download_log(
            "RAID",
            "Hugging Face liamdugan/raid",
            command,
            True,
            out_path,
            f"Saved shape {train_df.shape}. raid-bench was not used because it forces old sklearn/numpy builds on Python 3.14.",
        )
        print(f"Saved RAID train stream to {out_path}", flush=True)
    except Exception as exc:
        append_download_log(
            "RAID",
            "Hugging Face liamdugan/raid",
            command,
            False,
            raw_dir,
            f"{type(exc).__name__}: {exc}. Manual fallback: download RAID from Hugging Face/GitHub into data/raw/raid/.",
        )
        print("RAID download failed; see outputs/reports/dataset_download_log.md", flush=True)


def download_arabic_dataset(dataset_name: str) -> None:
    from datasets import load_dataset

    raw_dir = ROOT / "data" / "raw" / "arabic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = f"datasets.load_dataset('{dataset_name}')"
    try:
        dataset_dict = load_dataset(dataset_name)
        safe_name = dataset_name.split("/")[-1].replace("-", "_")
        saved = []
        for split, table in dataset_dict.items():
            df = table.to_pandas()
            out_path = raw_dir / f"{safe_name}_{split}.parquet"
            df.to_parquet(out_path, index=False)
            saved.append(out_path)
            print(f"Saved Arabic raw split: {out_path} shape={df.shape}", flush=True)
        append_download_log(dataset_name, f"Hugging Face {dataset_name}", command, True, raw_dir, f"Saved {len(saved)} files.")
    except Exception as exc:
        append_download_log(dataset_name, f"Hugging Face {dataset_name}", command, False, raw_dir, f"{type(exc).__name__}: {exc}")
        print(f"Arabic download failed for {dataset_name}; see outputs/reports/dataset_download_log.md", flush=True)


def main() -> None:
    args = parse_args()
    init_download_log()
    if not args.skip_semeval:
        download_semeval()
    if not args.skip_raid:
        download_raid(args.raid_max_scan)
    if not args.skip_arabic:
        download_arabic_dataset(ARABIC_ABSTRACTS)
        if args.include_arabic_social:
            download_arabic_dataset(ARABIC_SOCIAL)


if __name__ == "__main__":
    main()
