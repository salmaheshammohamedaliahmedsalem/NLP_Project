# Global AI vs Human Content Detection

Fresh restart workspace for a leakage-aware, paper-quality NLP project.

The previous implementation, experiments, paper draft, Streamlit app, and generated artifacts have been archived under `old/`. The new root is intentionally clean so the project can be rebuilt with a stronger design while preserving everything we learned.

## Current Direction

We will use prior findings instead of starting from zero:

- The original Kaggle-style dataset had severe leakage through explicit `AI-generated` markers and metadata such as `source` and `ai_model`.
- Cleaning that dataset reduced it from 20,000 rows to 6,406 usable text rows.
- Previous models trained on the original cleaned dataset failed to generalize to RAID, with best macro-F1 around 0.413 on the RAID quick subset.
- RAID is the stronger benchmark for the final paper because it supports generator-holdout, domain-holdout, attack-holdout, and decoding-strategy evaluation.

## New Project Goal

Build a robust AI-vs-human detector using RAID as the main benchmark, with:

- clean data ingestion and audit reports
- at least three related-work baselines on the same dataset
- proposed hybrid model with stylometric machine-signature features
- ablation study
- generator/domain/attack holdout evaluation
- IEEE-format technical report
- Streamlit demo after the modeling pipeline is stable

## Fresh Structure

```text
.
├── app/              # future Streamlit app
├── config/           # experiment configuration
├── data/
│   ├── raw/          # raw downloaded data or manifests
│   ├── processed/    # cleaned training/evaluation files
│   └── external/     # external benchmark subsets such as RAID
├── models/           # saved model artifacts
├── notebooks/        # exploratory notebooks only
├── old/              # archived previous implementation and outputs
├── paper/            # new IEEE paper source
├── reports/          # generated summaries and analysis notes
├── results/          # metrics, tables, logs, plots
├── scripts/          # runnable pipeline entry points
└── src/              # reusable package code
```

## Immediate Next Steps

1. Rebuild RAID data preparation with configurable sampling across multiple domains, models, and attacks.
2. Train three related-work baselines on the same RAID subset.
3. Add stylometric feature extraction and ablations.
4. Add the proposed hybrid model only after baselines are stable.
5. Write the new IEEE paper from the RAID-first results.

