# Dataset Download Log

## raid-bench package

- Source: PyPI raid-bench==0.2.0
- Command: `.venv/bin/python -m pip install -r requirements.txt with raid-bench==0.2.0`
- Status: failed
- Local raw path: `data/raw/raid`
- Notes/errors: Package forced old numpy/scikit-learn builds on Python 3.14 and failed during scikit-learn metadata/Cython build. Pipeline uses Hugging Face liamdugan/raid streaming fallback instead.

## SemEval-2024 Task 8 Subtask A

- Source: Google Drive folder 1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc
- Command: `gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc -O /Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/semeval`
- Status: failed
- Local raw path: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/semeval`
- Notes/errors: FileNotFoundError(2, 'No such file or directory')
Manual fallback: download SemEval-2024 Task 8 Subtask A from the official source and place files under data/raw/semeval/.

## RAID

- Source: Hugging Face liamdugan/raid
- Command: `datasets.load_dataset('liamdugan/raid', split='train', streaming=True), first 20000 rows`
- Status: failed
- Local raw path: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/raid`
- Notes/errors: RuntimeError: Cannot send a request, as the client has been closed.. Manual fallback: download RAID from Hugging Face/GitHub into data/raw/raid/.

## KFUPM-JRCAI/arabic-generated-abstracts

- Source: Hugging Face KFUPM-JRCAI/arabic-generated-abstracts
- Command: `datasets.load_dataset('KFUPM-JRCAI/arabic-generated-abstracts')`
- Status: failed
- Local raw path: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/arabic`
- Notes/errors: RuntimeError: Cannot send a request, as the client has been closed.

## RAID

- Source: Hugging Face liamdugan/raid
- Command: `datasets.load_dataset('liamdugan/raid', split='train', streaming=True), first 20000 rows`
- Status: succeeded
- Local raw path: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/raid/raid_train_stream_20000.parquet`
- Notes/errors: Saved shape (20000, 11). raid-bench was not used because it forces old sklearn/numpy builds on Python 3.14.

## KFUPM-JRCAI/arabic-generated-abstracts

- Source: Hugging Face KFUPM-JRCAI/arabic-generated-abstracts
- Command: `datasets.load_dataset('KFUPM-JRCAI/arabic-generated-abstracts')`
- Status: succeeded
- Local raw path: `/Users/salmaheshamsalem/Desktop/NLP_Project/data/raw/arabic`
- Notes/errors: Saved 3 files.

## SemEval-2024 Task 8 Subtask A

- Source: Google Drive folder 1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc
- Command: `.venv/bin/python -m gdown --folder https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc -O data/raw/semeval`
- Status: failed
- Local raw path: `data/raw/semeval`
- Notes/errors: Google Drive request timed out locally. Manual fallback required: download official SemEval-2024 Task 8 Subtask A files and place them under data/raw/semeval/, then run scripts/prepare_semeval.py and scripts/02_make_splits.py.

