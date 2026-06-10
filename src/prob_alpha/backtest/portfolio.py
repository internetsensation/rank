from __future__ import annotations

import pandas as pd


def next_period_returns(prices: pd.DataFrame, horizon: int = 21) -> pd.Series:
    """Compute forward returns for portfolio holding periods."""
    adj = prices["adj_close"].sort_index()
    future = adj.groupby(level="ticker").shift(-horizon)
    return (future / adj - 1.0).rename("asset_forward_return")


def portfolio_returns(weights: pd.Series, asset_returns: pd.Series) -> pd.Series:
    """Aggregate asset-level forward returns into portfolio returns by date."""
    aligned = pd.concat([weights.rename("weight"), asset_returns.rename("ret")], axis=1).dropna()
    return (aligned["weight"] * aligned["ret"]).groupby(level="date").sum().rename("strategy_return")


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Convert period returns into an equity curve."""
    return (initial_capital * (1.0 + returns.fillna(0.0)).cumprod()).rename("equity")
