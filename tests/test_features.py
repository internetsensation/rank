import pandas as pd

from prob_alpha.features.build_features import build_features


def sample_prices():
    dates = pd.date_range("2020-01-01", periods=320, freq="B")
    frames = []
    for ticker, offset in [("AAA", 0), ("BBB", 10)]:
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": range(100 + offset, 100 + offset + len(dates)),
                "high": range(101 + offset, 101 + offset + len(dates)),
                "low": range(99 + offset, 99 + offset + len(dates)),
                "close": range(100 + offset, 100 + offset + len(dates)),
                "adj_close": range(100 + offset, 100 + offset + len(dates)),
                "volume": 1_000_000,
            }
        )
        frames.append(df)
    return pd.concat(frames).set_index(["date", "ticker"])


def test_build_features_returns_expected_columns():
    features = build_features(sample_prices())
    assert "momentum_12_1" in features.columns
    assert "return_1m" in features.columns
    assert "vol_20d" in features.columns
    assert "dist_52w_high" in features.columns
    assert not features.empty
