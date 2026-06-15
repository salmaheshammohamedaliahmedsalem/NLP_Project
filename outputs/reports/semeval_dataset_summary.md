# SemEval Dataset Preparation Summary

- Status: not prepared
- Reason: no raw SemEval Subtask A files are available under `data/raw/semeval/`
- Required user action: manually download the official SemEval-2024 Task 8 Subtask A folder into `data/raw/semeval/` if Google Drive blocks `gdown`
- Next command after files are added: `.venv/bin/python scripts/prepare_semeval.py && .venv/bin/python scripts/02_make_splits.py`
