from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    total = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return float(total ** (1 / years) - 1) if years > 0 else float("nan")


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    return float(returns.dropna().std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12, risk_free_rate: float = 0.0) -> float:
    excess = returns.dropna() - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def summarize_returns(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    return {
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "n_periods": float(returns.dropna().shape[0]),
    }
