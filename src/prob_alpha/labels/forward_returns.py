from __future__ import annotations

import pandas as pd


def forward_returns(adj_close: pd.Series, horizon: int = 21) -> pd.Series:
    """Compute forward percentage returns by ticker."""
    future = adj_close.groupby(level="ticker").shift(-horizon)
    return future / adj_close - 1.0
