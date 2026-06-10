from __future__ import annotations

import pandas as pd


def to_datetime_index(values) -> pd.DatetimeIndex:
    """Convert an iterable of date-like values to a normalized DatetimeIndex."""
    return pd.to_datetime(values).tz_localize(None).normalize()


def month_end_rebalance_dates(dates) -> pd.DatetimeIndex:
    """Return available month-end dates from a collection of trading dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    if len(idx) == 0:
        return idx
    by_period = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    return pd.DatetimeIndex(by_period.values)
