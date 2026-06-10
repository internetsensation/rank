from __future__ import annotations

import pandas as pd

from prob_alpha.labels.forward_returns import forward_returns


def outperform_median_label(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Label whether each stock beats the same-date universe median forward return."""
    fwd = forward_returns(prices["adj_close"].sort_index(), horizon=horizon).rename("forward_return")
    median = fwd.groupby(level="date").transform("median")
    label = (fwd > median).astype(int).rename("outperform_median")
    out = pd.concat([fwd, median.rename("cross_sectional_median_return"), label], axis=1)
    return out.dropna()
