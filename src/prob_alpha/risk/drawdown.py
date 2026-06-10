from __future__ import annotations

import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns."""
    equity = (1 + returns.fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def drawdown_table(returns: pd.Series) -> pd.DataFrame:
    """Return a compact drawdown summary."""
    dd = drawdown_series(returns)
    return pd.DataFrame(
        {
            "max_drawdown": [dd.min()],
            "current_drawdown": [dd.iloc[-1] if len(dd) else 0.0],
        }
    )
