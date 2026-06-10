from __future__ import annotations

import pandas as pd

from prob_alpha.features.momentum import momentum_12_1, trailing_return
from prob_alpha.features.technical import distance_from_high, rolling_max_drawdown
from prob_alpha.features.volatility import realized_volatility, downside_volatility
from prob_alpha.features.volume import average_volume, volume_change


def build_features(prices: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Build the first cross-sectional equity feature set.

    Parameters
    ----------
    prices:
        MultiIndex DataFrame indexed by date/ticker with `adj_close` and `volume`.
    config:
        Feature config dictionary. If omitted, sensible baseline defaults are used.
    """
    config = config or {}
    feature_cfg = config.get("features", config)

    adj = prices["adj_close"].sort_index()
    vol = prices["volume"].sort_index()

    momentum_lookback = int(feature_cfg.get("momentum_lookback_days", 252))
    momentum_skip = int(feature_cfg.get("momentum_skip_days", 21))
    short_return_days = int(feature_cfg.get("short_return_days", 21))
    high_window = int(feature_cfg.get("high_window_days", 252))
    volume_windows = feature_cfg.get("volume_windows", [20, 60])
    drawdown_window = int(feature_cfg.get("drawdown_window_days", 252))

    features = pd.DataFrame(index=prices.index)
    features["momentum_12_1"] = momentum_12_1(adj, momentum_lookback, momentum_skip)
    features["return_1m"] = trailing_return(adj, short_return_days)

    for window in feature_cfg.get("volatility_windows", [20, 60]):
        window = int(window)
        features[f"vol_{window}d"] = realized_volatility(adj, window)

    features["downside_vol_20d"] = downside_volatility(adj, 20)
    features["dist_52w_high"] = distance_from_high(adj, high_window)
    features[f"avg_volume_{volume_windows[0]}d"] = average_volume(vol, int(volume_windows[0]))
    features["volume_change"] = volume_change(vol, int(volume_windows[0]), int(volume_windows[1]))
    features["max_drawdown_252d"] = rolling_max_drawdown(adj, drawdown_window)

    features = features.replace([float("inf"), float("-inf")], pd.NA)
    return features.dropna()
