from __future__ import annotations

import pandas as pd

from prob_alpha.backtest.costs import transaction_costs
from prob_alpha.backtest.metrics import summarize_returns
from prob_alpha.backtest.portfolio import equity_curve, next_period_returns, portfolio_returns
from prob_alpha.backtest.signals import top_fraction_signal


def run_top_decile_backtest(
    prices: pd.DataFrame,
    scores: pd.Series,
    horizon: int = 21,
    top_fraction: float = 0.10,
    cost_bps: float = 10.0,
    initial_capital: float = 100_000,
    min_names: int = 3,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a simple monthly-style top-fraction backtest.

    The caller should pass scores only on rebalance dates if monthly rebalancing is desired.
    """
    weights = top_fraction_signal(scores, top_fraction=top_fraction, min_names=min_names)
    asset_fwd_returns = next_period_returns(prices, horizon=horizon)
    gross = portfolio_returns(weights, asset_fwd_returns)
    costs = transaction_costs(weights, cost_bps=cost_bps).reindex(gross.index).fillna(0.0)
    net = (gross - costs).rename("net_return")
    out = pd.concat([gross, costs.rename("transaction_cost"), net], axis=1)
    out["equity"] = equity_curve(out["net_return"], initial_capital=initial_capital)
    metrics = summarize_returns(out["net_return"], periods_per_year=round(252 / horizon))
    metrics["avg_turnover"] = float(costs.mean() / (cost_bps / 10_000.0)) if cost_bps else 0.0
    return out, metrics
