from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit_app.utils import find_model_artifacts, load_model, predict_with_score, stylometric_summary


st.set_page_config(page_title="AI vs Human Detector", layout="wide")

st.title("Leakage-Aware AI vs Human Content Detection")
st.caption(
    "Text-only inference app. Metadata such as generator, source, domain, prompt, or dataset name is not used for prediction."
)

artifacts = find_model_artifacts()
if not artifacts:
    st.warning("No saved model artifacts found under outputs/models/. Run the experiment scripts first.")
else:
    selected = st.selectbox("Model artifact", artifacts, format_func=lambda path: str(path.relative_to(ROOT)))
    model = load_model(selected)
    text = st.text_area("Enter text", height=220)
    if st.button("Predict", type="primary") and text.strip():
        prediction, score = predict_with_score(model, text)
        st.subheader("Prediction")
        st.write("AI-generated" if prediction == 1 else "Human-written")
        if score is not None:
            st.write(f"Model score: `{score:.4f}`")
        else:
            st.write("Confidence/score is not available for this model.")
        st.subheader("Stylometric summary")
        st.dataframe(stylometric_summary(text), use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "Research framing: the old dataset is treated as a leakage-audit case study; "
    "SemEval and RAID are the main English benchmarks, and Arabic is a proof-of-concept."
)
st.markdown("Project source: https://github.com/salmaheshammohamedaliahmedsalem/NLP_Project")
