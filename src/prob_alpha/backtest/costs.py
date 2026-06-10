from __future__ import annotations

import pandas as pd


def turnover(weights: pd.DataFrame | pd.Series) -> pd.Series:
    """Compute one-way portfolio turnover from weights indexed by date/ticker."""
    if isinstance(weights, pd.Series):
        w = weights.unstack("ticker").fillna(0.0)
    else:
        w = weights.fillna(0.0)
    return w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1)) / 2.0


def transaction_costs(weights: pd.Series, cost_bps: float = 10.0) -> pd.Series:
    """Compute transaction costs as turnover times cost in basis points."""
    return turnover(weights) * (cost_bps / 10_000.0)
