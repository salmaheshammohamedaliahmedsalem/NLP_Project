# Fresh Start Plan

## What Was Archived

Everything from the previous project state was moved into `old/`, including:

- old Streamlit app
- old source package
- old scripts
- old results and logs
- old model artifacts
- old paper draft and references
- previous RAID quick experiments
- previous PDF source material

## Decisions Carried Forward

- Do not claim high scores from leaky data.
- Treat the original dataset as a leakage case study, not the main benchmark.
- Use RAID as the main serious benchmark.
- Report random split scores, but prioritize generator-holdout, domain-holdout, and attack-holdout scores.
- Keep stylometry because it supports explainability and the project topic.

## New Paper Story

Working title:

**Robust and Explainable AI-Generated Content Detection with RAID-Based Cross-Generator Evaluation**

Main claim:

AI-generated content detection should be evaluated under leakage-aware and cross-generator settings. Random split performance is not enough.

Planned sections:

- Abstract
- Introduction
- Related Work
- Dataset and Leakage Audit
- Methodology
- Comparative Study
- Proposed Hybrid Model
- Ablation Study
- Results and Discussion
- Limitations
- Conclusion

