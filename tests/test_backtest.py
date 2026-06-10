import pandas as pd

from prob_alpha.backtest.engine import run_top_decile_backtest


def test_run_top_decile_backtest_smoke():
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    rows = []
    for ticker, slope in [("AAA", 1.0), ("BBB", 0.8), ("CCC", 1.2), ("DDD", 0.6)]:
        for i, date in enumerate(dates):
            price = 100 + slope * i
            rows.append((date, ticker, price, price, price, price, price, 1000))
    prices = pd.DataFrame(
        rows,
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
    ).set_index(["date", "ticker"])

    score_dates = dates[::21][:3]
    idx = pd.MultiIndex.from_product([score_dates, ["AAA", "BBB", "CCC", "DDD"]], names=["date", "ticker"])
    scores = pd.Series(range(len(idx)), index=idx, name="p_outperform")
    result, metrics = run_top_decile_backtest(prices, scores, horizon=21, top_fraction=0.25, min_names=1)
    assert not result.empty
    assert "sharpe" in metrics
