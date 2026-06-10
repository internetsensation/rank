from __future__ import annotations

import pandas as pd

from prob_alpha.data.load_prices import PRICE_COLUMNS


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV prices into canonical research format."""
    df = prices.copy()
    if not isinstance(df.index, pd.MultiIndex):
        if {"date", "ticker"}.issubset(df.columns):
            df = df.set_index(["date", "ticker"])
        else:
            raise ValueError("prices must have MultiIndex(date, ticker) or date/ticker columns")

    df = df.sort_index()
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]), level=0)
    df = df[PRICE_COLUMNS]
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["adj_close"])
    df = df[df["adj_close"] > 0]
    df["volume"] = df["volume"].fillna(0)
    return df
