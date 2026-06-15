from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_human_detector.features import extract_one  # noqa: E402
from ai_human_detector.models import make_text_lr  # noqa: E402


GITHUB_URL = "https://github.com/salmaheshammohamedaliahmedsalem/NLP_Project"
DATA_PATH = ROOT / "data" / "processed" / "clean_ai_vs_human_content.csv"
RESULTS_PATH = ROOT / "results" / "experiment_results.csv"
HISTORY_PATH = ROOT / "results" / "training_history.csv"
TRAINING_LOG_PATH = ROOT / "results" / "training.log"
LOSS_CURVE_PATH = ROOT / "results" / "training_loss_curve.svg"
EPOCH_MODEL_PATH = ROOT / "models" / "epoch_tfidf_sgd.joblib"


st.set_page_config(
    page_title="AI vs Human Content Detector",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f7f4;
            color: #1d1d1b;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #deded8;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        .hero {
            border-bottom: 1px solid #d9d8d0;
            padding-bottom: 1.1rem;
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            font-size: 2.25rem;
            line-height: 1.1;
            margin: 0 0 0.35rem 0;
            color: #171716;
        }
        .hero p {
            font-size: 1rem;
            color: #55554d;
            margin: 0;
        }
        .verdict {
            background: #ffffff;
            border: 1px solid #d9d8d0;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .verdict strong {
            font-size: 1.2rem;
        }
        .small-note {
            color: #68685f;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=["content", "label_binary"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_training_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_training_log() -> str:
    if TRAINING_LOG_PATH.exists():
        return TRAINING_LOG_PATH.read_text(encoding="utf-8")
    return "Training log is not available yet. Run: python3 scripts/train_epoch_model.py --epochs 25"


@st.cache_resource(show_spinner="Training detector on cleaned research dataset...")
def train_detector() -> object:
    if EPOCH_MODEL_PATH.exists():
        return joblib.load(EPOCH_MODEL_PATH)
    df = load_training_data()
    model = make_text_lr()
    model.fit(df["content"], df["label_binary"])
    return model


def predict_probability_ai(model: object, text: str) -> float:
    if isinstance(model, dict) and {"vectorizer", "classifier"}.issubset(model):
        matrix = model["vectorizer"].transform([text])
        return float(model["classifier"].predict_proba(matrix)[0, 1])
    return float(model.predict_proba([text])[0, 1])


def probability_label(probability_ai: float) -> tuple[str, str]:
    if probability_ai >= 0.70:
        return "Likely AI-generated", "High confidence"
    if probability_ai >= 0.55:
        return "Possibly AI-generated", "Moderate confidence"
    if probability_ai <= 0.30:
        return "Likely human-written", "High confidence"
    if probability_ai <= 0.45:
        return "Possibly human-written", "Moderate confidence"
    return "Uncertain", "Low confidence"


def render_feature_panel(text: str) -> None:
    features = extract_one(text)
    feature_rows = [
        ("Words", int(features["word_count"])),
        ("Characters", int(features["char_count"])),
        ("Sentences", int(features["sentence_count"])),
        ("Lexical diversity", round(features["lexical_diversity"], 3)),
        ("Repeated bigram ratio", round(features["repeated_bigram_ratio"], 3)),
        ("Punctuation ratio", round(features["punctuation_ratio"], 3)),
        ("Average sentence length", round(features["avg_sentence_length"], 2)),
        ("Word entropy", round(features["word_entropy"], 2)),
    ]
    st.dataframe(
        pd.DataFrame(feature_rows, columns=["Signal", "Value"]),
        use_container_width=True,
        hide_index=True,
    )


def render_results_summary() -> None:
    results = load_results()
    if results.empty:
        st.info("Experiment result CSV is not available yet.")
        return
    display_cols = ["split", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    available_cols = [col for col in display_cols if col in results.columns]
    st.dataframe(
        results[available_cols].sort_values(["split", "f1"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


def stream_epoch_training(epochs: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_epoch_model.py"),
        "--epochs",
        str(epochs),
    ]
    output_box = st.empty()
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            lines.append(line.rstrip())
            output_box.code("\n".join(lines[-120:]), language="text")
    return_code = process.wait()
    if return_code != 0:
        st.error(f"Training failed with exit code {return_code}.")
    else:
        st.success("Training finished. Refreshing cached artifacts.")
        st.cache_data.clear()
        st.cache_resource.clear()


def main() -> None:
    inject_css()

    with st.sidebar:
        st.header("Platform")
        st.markdown(f"[GitHub repository]({GITHUB_URL})")
        st.markdown("[IEEE paper](paper/main.tex)")
        st.divider()
        st.caption("Model")
        if EPOCH_MODEL_PATH.exists():
            st.write("Epoch-trained TF-IDF + SGD logistic detector")
        else:
            st.write("TF-IDF word n-grams + Logistic Regression")
        st.caption("Training data")
        training_data = load_training_data()
        st.write(f"{len(training_data):,} strict-cleaned samples")
        st.write(training_data["label_binary"].map({0: "human", 1: "ai"}).value_counts().to_frame("count"))
        st.divider()
        st.caption("Important")
        st.write(
            "This is a leakage-aware research prototype, not a production plagiarism detector."
        )

    st.markdown(
        """
        <div class="hero">
          <h1>AI vs Human Content Detector</h1>
          <p>Research platform for detecting machine-written text using the cleaned Global AI vs Human Content 2026 dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Cleaned Samples", f"{len(load_training_data()):,}")
    with metric_b:
        st.metric("Removed Raw Rows", "13,594")
    with metric_c:
        st.metric("AI Marker Rows Removed", "10,004")

    tab_detect, tab_training, tab_experiments, tab_method = st.tabs(
        ["Detect", "Training Logs", "Experiment Results", "Method"]
    )

    with tab_detect:
        left, right = st.columns([1.25, 0.75], gap="large")
        with left:
            text = st.text_area(
                "Paste text or code to analyze",
                height=260,
                placeholder="Paste a paragraph, article excerpt, answer, or code snippet here...",
            )
            analyze = st.button("Analyze content", type="primary", use_container_width=True)

        with right:
            st.subheader("Detector Output")
            if analyze and text.strip():
                model = train_detector()
                probability_ai = predict_probability_ai(model, text)
                verdict, confidence = probability_label(probability_ai)
                probability_human = 1.0 - probability_ai
                st.markdown(
                    f"""
                    <div class="verdict">
                        <strong>{verdict}</strong><br>
                        <span class="small-note">{confidence}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(probability_ai, text=f"AI probability: {probability_ai:.1%}")
                st.progress(probability_human, text=f"Human probability: {probability_human:.1%}")
                st.caption("Scores are calibrated for this cleaned dataset and may shift on real-world text.")
            elif analyze:
                st.warning("Paste some text before analysis.")
            else:
                st.info("Enter text and run the detector.")

        if text.strip():
            st.subheader("Stylometric Signals")
            render_feature_panel(text)

    with tab_training:
        st.subheader("Training and Validation Loss")
        with st.expander("Run training from the UI", expanded=False):
            epochs = st.slider("Epochs", min_value=5, max_value=50, value=25, step=5)
            st.caption("This streams the same terminal logs produced by `scripts/train_epoch_model.py`.")
            if st.button("Run epoch training and show terminal", type="primary"):
                stream_epoch_training(epochs)

        history = load_training_history()
        if history.empty:
            st.warning("No training history found. Run `python3 scripts/train_epoch_model.py --epochs 25`.")
        else:
            chart_data = history.set_index("epoch")[["train_loss", "val_loss"]]
            st.line_chart(chart_data, use_container_width=True)
            if LOSS_CURVE_PATH.exists():
                st.image(str(LOSS_CURVE_PATH), caption="Saved training/validation loss diagram")
            best_row = history.sort_values(["val_f1", "val_roc_auc"], ascending=False).iloc[0]
            metric_1, metric_2, metric_3 = st.columns(3)
            with metric_1:
                st.metric("Best Epoch", int(best_row["epoch"]))
            with metric_2:
                st.metric("Best Validation F1", f"{best_row['val_f1']:.3f}")
            with metric_3:
                st.metric("Best Validation AUC", f"{best_row['val_roc_auc']:.3f}")
            st.dataframe(history, use_container_width=True, hide_index=True)

        st.subheader("Terminal Training Log")
        st.code(load_training_log(), language="text")
        st.caption("Re-run locally with: python3 scripts/train_epoch_model.py --epochs 25")

    with tab_experiments:
        st.subheader("Research Experiment Table")
        render_results_summary()

    with tab_method:
        st.subheader("Leakage-Aware Pipeline")
        st.write(
            "The platform trains only on cleaned content. It removes explicit AI markers, drops normalized duplicates, removes label conflicts, and excludes metadata fields such as source, AI model, prompt, length columns, and IDs."
        )
        st.write(
            "The paper and scripts report random, prompt-grouped, challenge, transformer embedding, and sanity-check results."
        )
        st.markdown(f"Project source: [{GITHUB_URL}]({GITHUB_URL})")


if __name__ == "__main__":
    main()
