from __future__ import annotations

import argparse

from prob_alpha.data.load_prices import download_yfinance_prices, save_prices
from prob_alpha.data.universe import from_config
from prob_alpha.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    universe = from_config(cfg)
    data_cfg = cfg["data"]
    prices = download_yfinance_prices(
        universe.all_symbols,
        start=data_cfg["start_date"],
        end=data_cfg.get("end_date"),
    )
    save_prices(prices, data_cfg["raw_prices_path"])
    save_prices(prices, data_cfg["processed_prices_path"])
    print(f"Saved {len(prices):,} price rows for {len(universe.all_symbols)} symbols.")


if __name__ == "__main__":
    main()
