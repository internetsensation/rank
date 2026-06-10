from __future__ import annotations

import numpy as np
import pandas as pd

from prob_alpha.backtest.metrics import summarize_returns


def bootstrap_return_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    path_length: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap return paths by resampling observed strategy returns."""
    rng = np.random.default_rng(random_seed)
    r = returns.dropna().to_numpy()
    if len(r) == 0:
        raise ValueError("returns must contain at least one non-null value")
    path_length = path_length or len(r)
    samples = rng.choice(r, size=(path_length, n_paths), replace=True)
    return pd.DataFrame(samples, columns=[f"path_{i}" for i in range(n_paths)])


def summarize_bootstrap_paths(paths: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """Summarize bootstrapped paths with performance metrics."""
    rows = []
    for col in paths:
        m = summarize_returns(paths[col], periods_per_year=periods_per_year)
        m["path"] = col
        rows.append(m)
    return pd.DataFrame(rows).set_index("path")
