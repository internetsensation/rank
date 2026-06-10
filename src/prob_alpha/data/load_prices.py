from __future__ import annotations

from pathlib import Path

import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def load_csv_prices(path: str | Path) -> pd.DataFrame:
    """Load prices from a CSV file into canonical MultiIndex format."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", *PRICE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.sort_values(["date", "ticker"]).set_index(["date", "ticker"])
    return df[PRICE_COLUMNS]


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    """Save canonical price data to CSV or Parquet based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.reset_index().to_csv(path, index=False)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load canonical price data from CSV or Parquet."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(["date", "ticker"])
        return df.sort_index()
    return load_csv_prices(path)


def download_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily prices from Yahoo Finance.

    yfinance is kept as an optional dependency because professional research should eventually
    replace this with a cleaner data vendor or a locally versioned dataset.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance with `pip install yfinance` to use this loader.") from exc

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames = []
    for ticker in tickers:
        if len(tickers) == 1:
            sub = raw.copy()
        else:
            sub = raw[ticker].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        sub["ticker"] = ticker.upper()
        sub.index.name = "date"
        frames.append(sub.reset_index())

    if not frames:
        raise ValueError("No price data downloaded.")

    df = pd.concat(frames, ignore_index=True)
    return df.set_index(["date", "ticker"])[PRICE_COLUMNS].sort_index()
