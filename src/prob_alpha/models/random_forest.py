from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_random_forest_pipeline(**params) -> Pipeline:
    """Create a random-forest classifier pipeline."""
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 25,
        "random_state": 42,
        "n_jobs": -1,
    }
    defaults.update(params)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**defaults)),
        ]
    )
