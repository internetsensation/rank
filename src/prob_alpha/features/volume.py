from __future__ import annotations

import pandas as pd


def average_volume(volume: pd.Series, window: int) -> pd.Series:
    """Compute trailing average share volume."""
    return volume.groupby(level="ticker").rolling(window).mean().droplevel(0)


def volume_change(volume: pd.Series, short_window: int = 20, long_window: int = 60) -> pd.Series:
    """Compute ratio of short-window average volume to long-window average volume."""
    short = average_volume(volume, short_window)
    long = average_volume(volume, long_window)
    return short / long - 1.0
