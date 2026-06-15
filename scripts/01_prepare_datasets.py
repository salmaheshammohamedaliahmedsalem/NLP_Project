from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all dataset preparation scripts.")
    parser.add_argument("--skip-semeval", action="store_true")
    parser.add_argument("--skip-raid", action="store_true")
    parser.add_argument("--skip-arabic", action="store_true")
    parser.add_argument("--raid-max-human", type=int, default=10_000)
    parser.add_argument("--raid-max-ai", type=int, default=10_000)
    parser.add_argument("--raid-allow-hf-fallback", action="store_true")
    parser.add_argument("--arabic-include-social", action="store_true")
    parser.add_argument("--arabic-skip-hf", action="store_true")
    return parser.parse_args()


def run_step(name: str, command: list[str]) -> bool:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        print(f"{name} failed with exit code {completed.returncode}. Continue to next dataset.")
        return False
    return True


def main() -> None:
    args = parse_args()
    python = sys.executable
    results: dict[str, bool] = {}

    if not args.skip_semeval:
        results["semeval"] = run_step("Prepare SemEval", [python, "scripts/prepare_semeval.py"])
    if not args.skip_raid:
        raid_command = [
            python,
            "scripts/prepare_raid.py",
            "--max-human",
            str(args.raid_max_human),
            "--max-ai",
            str(args.raid_max_ai),
        ]
        if args.raid_allow_hf_fallback:
            raid_command.append("--allow-hf-fallback")
        results["raid"] = run_step("Prepare RAID", raid_command)
    if not args.skip_arabic:
        arabic_command = [python, "scripts/prepare_arabic.py"]
        if args.arabic_include_social:
            arabic_command.append("--include-social")
        if args.arabic_skip_hf:
            arabic_command.append("--skip-hf")
        results["arabic"] = run_step("Prepare Arabic POC", arabic_command)

    print("\n=== Preparation summary ===")
    for dataset_name, ok in results.items():
        print(f"{dataset_name}: {'ok' if ok else 'failed/missing raw data'}")
    if results and not any(results.values()):
        raise SystemExit("No datasets were prepared. Check raw data paths and download logs.")


if __name__ == "__main__":
    main()
