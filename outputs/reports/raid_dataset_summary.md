# RAID Dataset Preparation Summary

RAID labels are derived from `model == human` when available; all other models are AI-generated.
Metadata is preserved for analysis but must not be used as model input.

- Raw shape: (20000, 11)
- Clean balanced shape: (3532, 7)
- Output: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/processed/raid_english_clean.csv`
- Class distribution: {0: 1766, 1: 1766}
- Domains: {'abstracts': 3532}
- Generators: {'human': 1766, 'llama-chat': 325, 'mistral-chat': 302, 'mpt-chat': 294, 'mistral': 293, 'gpt2': 278, 'mpt': 274}
- Attacks: {'none': 3532}
- Decoding values: {'sampling': 896, 'greedy': 870}
- Columns: ['text', 'label', 'dataset_name', 'domain', 'generator', 'attack', 'decoding']
