from __future__ import annotations

import pandas as pd


def trailing_return(adj_close: pd.Series, window: int) -> pd.Series:
    """Compute trailing percentage return over a rolling window."""
    return adj_close.groupby(level="ticker").pct_change(window)


def momentum_12_1(adj_close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Compute 12-1 momentum: return from t-lookback to t-skip."""
    shifted = adj_close.groupby(level="ticker").shift(skip)
    base = adj_close.groupby(level="ticker").shift(lookback)
    return shifted / base - 1.0
