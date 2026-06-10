from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class QuantileModelSet:
    """Fit one quantile regressor per requested quantile."""

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    params: dict | None = None

    def __post_init__(self):
        self.models_: dict[float, Pipeline] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        params = self.params or {}
        for q in self.quantiles:
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingRegressor(loss="quantile", alpha=q, random_state=42, **params),
                    ),
                ]
            )
            model.fit(X, y)
            self.models_[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise ValueError("QuantileModelSet must be fit before prediction.")
        preds = {f"q_{q:g}": model.predict(X) for q, model in self.models_.items()}
        return pd.DataFrame(preds, index=X.index)
