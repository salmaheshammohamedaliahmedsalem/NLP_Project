# Global AI vs. Human Content Detection 2026

This repository contains a reproducible NLP research pipeline for binary classification of human-written versus AI-generated content. The project is structured for a paper-style evaluation rather than a single notebook score.

## Research Focus

The dataset has severe leakage risks: metadata columns such as `source` and `ai_model` perfectly reveal the label, raw AI rows contain explicit `AI-generated` markers, and many normalized texts are duplicated or conflict across labels. The pipeline therefore treats data cleaning, leakage auditing, and strict split design as first-class experimental components.

## Project Structure

```text
.
├── ai_vs_human_content_v2_20000.csv
├── challenge_test.csv
├── data/processed/clean_ai_vs_human_content.csv
├── app.py
├── paper/main.tex
├── paper/references.bib
├── results/
├── scripts/run_experiments.py
├── scripts/run_transformer_embeddings.py
└── src/ai_human_detector/
```

## Run the Main Experiments

```bash
python3 scripts/run_experiments.py
```

## Run the Detection Platform

```bash
streamlit run app.py
```

The Streamlit interface provides live AI-vs-human prediction, AI/human probability bars, stylometric signal inspection, experiment result tables, and a direct GitHub link:

https://github.com/salmaheshammohamedaliahmedsalem/NLP_Project

This runs:

- TF-IDF word Logistic Regression
- TF-IDF character Logistic Regression
- TF-IDF word Linear SVM
- Stylometric Logistic Regression
- Hybrid TF-IDF + stylometric Logistic Regression

It evaluates models on:

- random stratified split
- prompt-grouped split
- external challenge test set
- sanity checks: shuffled labels, length-only, prompt-only

## Run Transformer Embedding Experiments

```bash
python3 scripts/run_transformer_embeddings.py
```

This uses cached `distilroberta-base` weights as a frozen encoder and trains a logistic classifier on transformer embeddings. It does not require internet access if the model is already cached.

## Key Outputs

- `results/metadata_leakage_report.csv`: metadata proxy leakage analysis
- `results/cleaning_audit.json`: cleaning and de-duplication summary
- `results/near_duplicate_audit.csv`: high-similarity text pair warnings
- `results/experiment_results.csv`: main sklearn experiment table
- `results/sanity_checks.csv`: leakage sanity checks
- `results/transformer_embedding_results.csv`: optional transformer results
- `paper/main.tex`: IEEE-style Overleaf-ready paper

## Important Finding

After strict cleaning, the dataset shrinks from 20,000 rows to 6,406 rows. All remaining clean samples are text samples because the code subset contains many conflicting normalized duplicates. This means strong claims about code-domain generalization are not scientifically supported by the cleaned local dataset without obtaining a better code split.
