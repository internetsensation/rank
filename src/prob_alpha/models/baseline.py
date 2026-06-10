from __future__ import annotations

import pandas as pd


class MedianProbabilityBaseline:
    """Baseline that predicts the unconditional positive-class frequency."""

    def __init__(self):
        self.p_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.p_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame):
        if self.p_ is None:
            raise ValueError("Model must be fit before prediction.")
        return pd.DataFrame({0: 1 - self.p_, 1: self.p_}, index=X.index)
