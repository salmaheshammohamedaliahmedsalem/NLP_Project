# Global AI vs. Human Content Detection 2026

This repository contains a reproducible NLP research pipeline for binary classification of human-written versus AI-generated content. The project is structured for a paper-style evaluation rather than a single notebook score.

## Streamlit Platform

- GitHub repository: https://github.com/salmaheshammohamedaliahmedsalem/NLP_Project
- Streamlit deploy link: https://share.streamlit.io/deploy?repository=salmaheshammohamedaliahmedsalem%2FNLP_Project&branch=main&mainModule=app.py
- App file on GitHub: https://github.com/salmaheshammohamedaliahmedsalem/NLP_Project/blob/main/app.py
- Local app command: `streamlit run app.py`

After Streamlit Cloud finishes deployment, replace the deploy link above with the generated `*.streamlit.app` URL.

## Research Focus

The dataset has severe leakage risks: metadata columns such as `source` and `ai_model` perfectly reveal the label, raw AI rows contain explicit `AI-generated` markers, and many normalized texts are duplicated or conflict across labels. The pipeline therefore treats data cleaning, leakage auditing, and strict split design as first-class experimental components.

## Project Structure

```text
.
├── ai_vs_human_content_v2_20000.csv
├── challenge_test.csv
├── data/processed/clean_ai_vs_human_content.csv
├── app.py
├── .streamlit/config.toml
├── models/epoch_tfidf_sgd.joblib
├── paper/main.tex
├── paper/references.bib
├── results/
├── reports/project_summary.md
├── scripts/run_experiments.py
├── scripts/run_transformer_embeddings.py
├── scripts/train_epoch_model.py
└── src/ai_human_detector/
```

## Run the Main Experiments

```bash
python3 scripts/run_experiments.py
```

## Run RAID Robustness Experiments

```bash
python3 scripts/prepare_raid_subset.py --output data/raid/raid_quick_subset.csv --max-human 1200 --max-ai 1200 --max-per-domain-label 500 --max-per-model 300 --max-scan 80000
python3 scripts/train_raid.py --data data/raid/raid_quick_subset.csv
python3 scripts/evaluate_previous_on_raid.py
```

RAID results are stored in:

- `data/raid/raid_quick_subset.csv`
- `results/raid/raid_training.log`
- `results/raid/raid_results.csv`
- `models/raid_best_tfidf.joblib`
- `results/raid/previous_models_on_raid.csv`
- `results/raid/previous_models_on_raid.log`

Current RAID quick-subset result:

- Random split best macro-F1: `0.980` using TF-IDF Word SVM.
- Source-grouped split best macro-F1: `0.964` using TF-IDF Word SVM.
- Held-out `llama-chat` model macro-F1 drops to `0.592` using TF-IDF Char LR.

This is a strong paper result because it shows that random scores can be high while unseen-generator robustness remains difficult.

External benchmark result for the previous project models:

- Previous models were trained on `data/processed/clean_ai_vs_human_content.csv`.
- They were tested directly on `data/raid/raid_quick_subset.csv`.
- Best previous-model macro-F1 on RAID: `0.413` using TF-IDF Char LR.
- Most previous models collapse toward one class on RAID.

This is the strongest evidence that the original dataset is not enough by itself and that RAID should be used as the serious benchmark.

## Run the Detection Platform

```bash
streamlit run app.py
```

The Streamlit interface provides live AI-vs-human prediction, AI/human probability bars, stylometric signal inspection, experiment result tables, training logs, and a direct GitHub link.

## Run Epoch Training with Terminal Logs

```bash
python3 scripts/train_epoch_model.py --epochs 25
```

This prints epoch-by-epoch terminal logs and writes:

- `results/training.log`
- `results/training_history.csv`
- `results/training_loss_curve.svg`
- `models/epoch_tfidf_sgd.joblib`

The Streamlit app shows the training/validation loss diagram and the terminal log in the `Training Logs` tab.

## Deploy on Streamlit Community Cloud

Use this prefilled deployment URL:

https://share.streamlit.io/deploy?repository=salmaheshammohamedaliahmedsalem%2FNLP_Project&branch=main&mainModule=app.py

Use these settings:

- Repository: `salmaheshammohamedaliahmedsalem/NLP_Project`
- Branch: `main`
- Main file path: `app.py`
- Python dependencies: `requirements.txt`
- Streamlit config: `.streamlit/config.toml`

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
- `results/raid/raid_results.csv`: RAID benchmark results
- `results/sanity_checks.csv`: leakage sanity checks
- `results/transformer_embedding_results.csv`: optional transformer results
- `paper/main.tex`: IEEE-style Overleaf-ready paper

## Important Finding

After strict cleaning, the dataset shrinks from 20,000 rows to 6,406 rows. All remaining clean samples are text samples because the code subset contains many conflicting normalized duplicates. This means strong claims about code-domain generalization are not scientifically supported by the cleaned local dataset without obtaining a better code split.
