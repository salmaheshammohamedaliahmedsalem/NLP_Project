# Global AI vs Human Content Detection 2026

Clean research pipeline for binary AI-generated content detection.

The project compares lexical, character-level, stylometric, and hybrid feature representations across two English datasets, then adds a small Arabic proof-of-concept.

## Research Questions

1. Which model performs best within each English benchmark?
2. Which dataset gives better generalization?
3. Does a detector trained on one dataset generalize to another?
4. Can the same lexical/stylometric approach extend to Arabic?

## Datasets

Main English datasets:

- SemEval-2024 Task 8 English
- RAID English subset

Arabic:

- Proof-of-concept only. Use SemEval Arabic or another Arabic human-vs-AI dataset if available.

## Models

- `M1_Word_TFIDF_LogReg`: word TF-IDF unigrams/bigrams + Logistic Regression
- `M2_Char_TFIDF_LinearSVM`: character TF-IDF 3-5 grams + Linear SVM
- `M3_Stylometric_RandomForest`: handcrafted stylometric features + Random Forest
- `M4_Hybrid_TFIDF_Stylometric`: word TF-IDF + character TF-IDF + stylometric features + Linear SVM

Metadata such as `generator`, `attack`, `domain`, `source`, `prompt`, and `dataset_name` is never used as model input.

## Directory Structure

```text
NLP_Project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── semeval/
│   │   ├── raid/
│   │   └── arabic/
│   ├── processed/
│   └── splits/
├── notebooks/
├── src/
├── outputs/
│   ├── models/
│   ├── results/
│   ├── figures/
│   └── reports/
├── streamlit_app/
└── old/
```

`old/` contains the previous project version and is kept only for reference.

## How To Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Put raw files in:

```text
data/raw/semeval/
data/raw/raid/
data/raw/arabic/
```

Prepare clean datasets:

```bash
python3 scripts/01_prepare_datasets.py --skip-raid-stream
```

If you want to stream a RAID subset from Hugging Face:

```bash
python3 scripts/01_prepare_datasets.py --raid-max-human 5000 --raid-max-ai 5000
```

Create train/test splits:

```bash
python3 scripts/02_make_splits.py
```

Run within-dataset experiments:

```bash
python3 scripts/03_run_english_experiments.py --dataset semeval
python3 scripts/03_run_english_experiments.py --dataset raid
```

Run cross-dataset evaluation:

```bash
python3 scripts/04_cross_dataset_evaluation.py
```

Run Arabic POC:

```bash
python3 scripts/05_arabic_poc.py
```

Build final tables:

```bash
python3 scripts/06_build_reports.py
```

Run the app after models are saved:

```bash
streamlit run app.py
```

## Outputs

- `outputs/results/semeval_results.csv`
- `outputs/results/raid_results.csv`
- `outputs/results/cross_dataset_results.csv`
- `outputs/results/arabic_poc_results.csv`
- `outputs/figures/confusion_matrices/`
- `outputs/figures/comparison_plots/`
- `outputs/figures/feature_importance/`
- `outputs/reports/experiment_summary.md`

## Final Research Claim

This project compares lexical, character-level, stylometric, and hybrid feature representations for AI-generated text detection across two English datasets. The proposed hybrid model combines TF-IDF lexical signals with stylometric machine-signature features. Cross-dataset evaluation is used to test whether model performance generalizes beyond a single benchmark. Finally, an Arabic proof-of-concept is included to examine whether the same detection approach can be extended to non-English text.
