from __future__ import annotations

import pandas as pd


class AverageProbabilityEnsemble:
    """Average predicted positive-class probabilities from multiple classifiers."""

    def __init__(self, models: list):
        self.models = models

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = [model.predict_proba(X)[:, 1] for model in self.models]
        p = sum(probs) / len(probs)
        return pd.DataFrame({0: 1 - p, 1: p}, index=X.index)
