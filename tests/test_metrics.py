import pandas as pd

from prob_alpha.backtest.metrics import max_drawdown, sharpe_ratio, summarize_returns


def test_metrics_are_finite_for_nonconstant_returns():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    summary = summarize_returns(returns)
    assert "sharpe" in summary
    assert max_drawdown(returns) <= 0
    assert sharpe_ratio(returns) == summary["sharpe"]
