from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_pipeline(**params) -> Pipeline:
    """Create a robust logistic-regression classifier pipeline."""
    defaults = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
    }
    defaults.update(params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**defaults)),
        ]
    )


def predict_outperformance_probability(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities as a Series aligned to X."""
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=X.index, name="p_outperform")
