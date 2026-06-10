from __future__ import annotations

import pandas as pd


def simple_triple_barrier_label(
    adj_close: pd.Series,
    horizon: int = 21,
    profit_take: float = 0.05,
    stop_loss: float = -0.05,
) -> pd.Series:
    """Simple triple-barrier style label.

    Returns:
    - 1 if profit-take barrier is hit first
    - -1 if stop-loss barrier is hit first
    - 0 if neither barrier is hit before the vertical horizon
    """
    labels = []
    names = []
    for ticker, s in adj_close.groupby(level="ticker"):
        s = s.droplevel("ticker").sort_index()
        for i, date in enumerate(s.index[:-horizon]):
            start = s.iloc[i]
            path = s.iloc[i + 1 : i + horizon + 1] / start - 1.0
            pt_hits = path[path >= profit_take]
            sl_hits = path[path <= stop_loss]
            if pt_hits.empty and sl_hits.empty:
                value = 0
            elif sl_hits.empty or (not pt_hits.empty and pt_hits.index[0] < sl_hits.index[0]):
                value = 1
            else:
                value = -1
            names.append((date, ticker))
            labels.append(value)
    return pd.Series(labels, index=pd.MultiIndex.from_tuples(names, names=["date", "ticker"]), name="triple_barrier")
