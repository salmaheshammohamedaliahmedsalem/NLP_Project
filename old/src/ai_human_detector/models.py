from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from ai_human_detector.features import extract_stylometric_frame


def _identity_text(values):
    return values


def make_text_lr() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def make_char_lr() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=60000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def make_text_svm() -> Pipeline:
    base = LinearSVC(class_weight="balanced", random_state=42)
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            ("clf", CalibratedClassifierCV(base, cv=3)),
        ]
    )


class StylometricOnlyModel:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    def fit(self, texts, labels):
        features = extract_stylometric_frame(texts)
        matrix = self.scaler.fit_transform(features)
        self.classifier.fit(matrix, labels)
        return self

    def predict(self, texts):
        features = extract_stylometric_frame(texts)
        return self.classifier.predict(self.scaler.transform(features))

    def predict_proba(self, texts):
        features = extract_stylometric_frame(texts)
        return self.classifier.predict_proba(self.scaler.transform(features))


class HybridTfidfStylometricModel:
    def __init__(self) -> None:
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        max_features=50000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=60000,
                    ),
                ),
            ]
        )
        self.scaler = StandardScaler(with_mean=False)
        self.sty_scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=42)

    def _matrix(self, texts, fit: bool):
        tfidf = self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        sty = extract_stylometric_frame(texts)
        sty_matrix = self.sty_scaler.fit_transform(sty) if fit else self.sty_scaler.transform(sty)
        return hstack([tfidf, sty_matrix], format="csr")

    def fit(self, texts, labels):
        self.classifier.fit(self._matrix(texts, fit=True), labels)
        return self

    def predict(self, texts):
        return self.classifier.predict(self._matrix(texts, fit=False))

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self._matrix(texts, fit=False))


def model_registry() -> dict[str, object]:
    return {
        "TF-IDF Word LR": make_text_lr(),
        "TF-IDF Char LR": make_char_lr(),
        "TF-IDF Word SVM": make_text_svm(),
        "Stylometric LR": StylometricOnlyModel(),
        "Hybrid TF-IDF + Stylometric LR": HybridTfidfStylometricModel(),
    }


def predict_scores(model, texts) -> tuple[np.ndarray, np.ndarray | None]:
    predictions = model.predict(texts)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
    else:
        scores = None
    return predictions, scores
