import pandas as pd

from prob_alpha.labels.outperform import outperform_median_label


def test_outperform_median_label_binary():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    frames = []
    for ticker, slope in [("AAA", 1.0), ("BBB", 2.0), ("CCC", 0.5)]:
        prices = 100 + slope * pd.Series(range(len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": prices.values,
                    "open": prices.values,
                    "high": prices.values,
                    "low": prices.values,
                    "close": prices.values,
                    "volume": 1000,
                }
            )
        )
    df = pd.concat(frames).set_index(["date", "ticker"])
    labels = outperform_median_label(df, horizon=5)
    assert set(labels["outperform_median"].unique()).issubset({0, 1})
    assert "forward_return" in labels.columns
