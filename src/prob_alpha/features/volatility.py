from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(adj_close: pd.Series) -> pd.Series:
    """Compute daily adjusted-close returns by ticker."""
    return adj_close.groupby(level="ticker").pct_change()


def realized_volatility(adj_close: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Compute trailing realized volatility."""
    returns = daily_returns(adj_close)
    vol = returns.groupby(level="ticker").rolling(window).std().droplevel(0)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def downside_volatility(adj_close: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Compute trailing downside volatility using negative daily returns only."""
    returns = daily_returns(adj_close)
    downside = returns.where(returns < 0, 0.0)
    vol = downside.groupby(level="ticker").rolling(window).std().droplevel(0)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
