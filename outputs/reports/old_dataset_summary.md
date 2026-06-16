# Old Dataset Preparation Summary

This benchmark uses the previous project CSV as a separate comparison dataset. It is not mixed with SemEval or RAID.

- Source: `old/ai_vs_human_content_v2_20000.csv`
- Rows before cleaning: 20000
- Missing text: 0
- Missing labels: 0
- Duplicate texts removed: 11758
- Final rows: 8242
- Class distribution: {0: 4075, 1: 4167}
- Content types: {'text': 8192, 'code': 50}
- Languages: {'en': 8192, 'python': 20, 'cpp': 20, 'java': 10}
- Final columns: ['text', 'label', 'dataset_name', 'type', 'source', 'topic', 'word_count', 'char_count', 'ai_model', 'language', 'complexity_score', 'is_multiline_code']
