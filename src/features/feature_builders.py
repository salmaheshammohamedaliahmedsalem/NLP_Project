from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from src.features.stylometric_features import extract_stylometric_features


class StylometricTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_stylometric_features(X).values
