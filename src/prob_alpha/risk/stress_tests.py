from __future__ import annotations

import pandas as pd


def apply_cost_shock(returns: pd.Series, extra_cost_per_period: float) -> pd.Series:
    """Subtract an additional cost from every period return."""
    return (returns - extra_cost_per_period).rename(f"cost_shock_{extra_cost_per_period:g}")


def apply_signal_decay(returns: pd.Series, decay: float) -> pd.Series:
    """Shrink returns toward zero to simulate weaker signal quality."""
    if not 0 <= decay <= 1:
        raise ValueError("decay must be in [0, 1]")
    return (returns * (1 - decay)).rename(f"signal_decay_{decay:g}")


def apply_volatility_spike(returns: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Magnify deviations from average return to simulate higher volatility."""
    mean = returns.mean()
    return (mean + multiplier * (returns - mean)).rename(f"vol_spike_{multiplier:g}")
