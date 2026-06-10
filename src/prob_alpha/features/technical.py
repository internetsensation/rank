from __future__ import annotations

import numpy as np
import pandas as pd


def distance_from_high(adj_close: pd.Series, window: int = 252) -> pd.Series:
    """Distance from trailing high. Values are <= 0."""
    high = adj_close.groupby(level="ticker").rolling(window).max().droplevel(0)
    return adj_close / high - 1.0


def _max_drawdown_window(values: np.ndarray) -> float:
    running_high = np.maximum.accumulate(values)
    drawdowns = values / running_high - 1.0
    return float(drawdowns.min())


def rolling_max_drawdown(adj_close: pd.Series, window: int = 252) -> pd.Series:
    """Trailing max drawdown over each rolling price window."""
    return (
        adj_close.groupby(level="ticker")
        .rolling(window)
        .apply(_max_drawdown_window, raw=True)
        .droplevel(0)
    )
