from __future__ import annotations

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE
from src.features.feature_builders import StylometricTransformer


class HybridTfidfStylometricSVM:
    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50000,
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
        )
        self.stylo = StylometricTransformer()
        self.scaler = StandardScaler()
        self.classifier = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)

    def fit(self, X, y):
        word = self.word_vectorizer.fit_transform(X)
        char = self.char_vectorizer.fit_transform(X)
        stylo = self.scaler.fit_transform(self.stylo.transform(X))
        features = hstack([word, char, stylo], format="csr")
        self.classifier.fit(features, y)
        return self

    def _transform(self, X):
        word = self.word_vectorizer.transform(X)
        char = self.char_vectorizer.transform(X)
        stylo = self.scaler.transform(self.stylo.transform(X))
        return hstack([word, char, stylo], format="csr")

    def predict(self, X):
        return self.classifier.predict(self._transform(X))

    def decision_function(self, X):
        return self.classifier.decision_function(self._transform(X))


def make_model(model_name: str):
    if model_name == "M1_Word_TFIDF_LogReg":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=50000,
                        stop_words="english",
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "M2_Char_TFIDF_LinearSVM":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        max_features=50000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "M3_Stylometric_RandomForest":
        return Pipeline(
            [
                ("stylo", StylometricTransformer()),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")),
            ]
        )
    if model_name == "M4_Hybrid_TFIDF_Stylometric":
        return HybridTfidfStylometricSVM()
    raise ValueError(f"Unknown model name: {model_name}")


def make_all_models(model_names: tuple[str, ...]):
    return {model_name: make_model(model_name) for model_name in model_names}
