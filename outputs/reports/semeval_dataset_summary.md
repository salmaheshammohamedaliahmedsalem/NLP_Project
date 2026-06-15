# SemEval Dataset Preparation Summary

Labels are mapped only from explicit binary human/AI values. Unknown labels raise an error.

- Final rows: 124552
- Final class distribution: {0: 65646, 1: 58906}
- Final columns: ['text', 'label', 'dataset_name', 'source_file']

## Source Files

### data/raw/semeval/subtaskA_dev_monolingual.jsonl
- Rows before cleaning: 5000
- Missing text: 0
- Missing labels: 0
- Duplicate texts removed: 47
- Rows after cleaning: 4953
- Class distribution: {0: 2453, 1: 2500}

### data/raw/semeval/subtaskA_train_monolingual.jsonl
- Rows before cleaning: 119757
- Missing text: 0
- Missing labels: 0
- Duplicate texts removed: 158
- Rows after cleaning: 119599
- Class distribution: {0: 63193, 1: 56406}
