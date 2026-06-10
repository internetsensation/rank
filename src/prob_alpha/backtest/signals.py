from __future__ import annotations

import pandas as pd


def top_fraction_signal(scores: pd.Series, top_fraction: float = 0.10, min_names: int = 1) -> pd.Series:
    """Create equal-weight long signals for top-ranked names by date."""
    if not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("scores must be indexed by date/ticker")

    def select(group: pd.Series) -> pd.Series:
        n = max(min_names, int(len(group) * top_fraction))
        selected = group.sort_values(ascending=False).head(n).index
        weights = pd.Series(0.0, index=group.index)
        if len(selected) > 0:
            weights.loc[selected] = 1.0 / len(selected)
        return weights

    return scores.groupby(level="date", group_keys=False).apply(select).rename("weight")


def uncertainty_adjusted_score(
    predicted_return: pd.Series,
    model_uncertainty: pd.Series,
    downside_risk: pd.Series,
    uncertainty_penalty: float = 1.5,
    downside_penalty: float = 0.5,
) -> pd.Series:
    """Rank using predicted return minus uncertainty and downside-risk penalties."""
    return (
        predicted_return
        - uncertainty_penalty * model_uncertainty
        - downside_penalty * downside_risk
    ).rename("uncertainty_adjusted_score")
