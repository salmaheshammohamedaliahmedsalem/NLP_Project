from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DOWNLOAD_LOG_PATH = REPORTS_DIR / "dataset_download_log.md"

LABEL_HUMAN = 0
LABEL_AI = 1

ENGLISH_DATASETS = ("semeval", "raid", "old_ai")
MODEL_NAMES = (
    "M1_Word_TFIDF_LogReg",
    "M2_Char_TFIDF_LinearSVM",
    "M3_Stylometric_RandomForest",
    "M4_Hybrid_TFIDF_Stylometric",
)
