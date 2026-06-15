# Project Summary

## What Changed

- Added a reusable Python package under `src/ai_human_detector`.
- Added a full sklearn experiment runner in `scripts/run_experiments.py`.
- Added a cached transformer embedding runner in `scripts/run_transformer_embeddings.py`.
- Generated a cleaned dataset at `data/processed/clean_ai_vs_human_content.csv`.
- Generated fresh audit and experiment outputs in `results/`.
- Added an IEEE-style Overleaf paper in `paper/main.tex` with references in `paper/references.bib`.

## Main Audit Finding

The original dataset is not safe for naive modeling:

- `source` and `ai_model` perfectly reveal the label.
- 10,004 rows contain explicit `AI-generated` markers.
- 918 normalized texts have conflicting labels.
- 1,121 normalized duplicate rows remain before strict duplicate removal.
- Strict cleaning reduces the dataset from 20,000 rows to 6,406 rows.

## Main Results

Best local sklearn results after strict cleaning:

- Random split: `TF-IDF Word SVM`, F1 = 0.602.
- Prompt-grouped split: `TF-IDF Word SVM`, F1 = 0.636.
- Challenge test: `TF-IDF Word LR`, F1 = 0.783.

Transformer embedding results with cached `distilroberta-base` and a 3,000-row training cap:

- Random split embedding LR: F1 = 0.521.
- Prompt-grouped embedding + stylometry: F1 = 0.550.
- Challenge embedding LR: F1 = 0.125.

## Interpretation

The honest conclusion is that the raw dataset can create fake perfect performance, but the cleaned dataset is much harder. This is defensible for a paper because it shows scientific handling of leakage instead of reporting inflated scores.
