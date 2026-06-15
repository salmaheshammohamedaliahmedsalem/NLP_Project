# Arabic POC Dataset Preparation Summary

Arabic text is lightly normalized: diacritics and tatweel are removed, Alef variants are normalized,
`ى` is normalized to `ي`, and whitespace is collapsed. Punctuation and casing are otherwise preserved.

- Final shape: (36523, 7)
- Output: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/processed/arabic_poc_clean.csv`
- Class distribution: {0: 2992, 1: 33531}
- Domains: {'academic_abstract': 36523}
- Generators: {'allam': 8388, 'openai': 8388, 'jais': 8381, 'llama': 8374, 'human': 2992}
- Source columns: {'allam_generated_abstract': 8388, 'openai_generated_abstract': 8388, 'jais_generated_abstract': 8381, 'llama_generated_abstract': 8374, 'original_abstract': 2992}
- Columns: ['text', 'label', 'dataset_name', 'domain', 'generator', 'source_column', 'source_file']
