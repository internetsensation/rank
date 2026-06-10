from __future__ import annotations

import argparse

import pandas as pd

from prob_alpha.backtest.engine import run_top_decile_backtest
from prob_alpha.data.load_prices import load_prices
from prob_alpha.utils.config import ensure_parent_dir, load_experiment_config
from prob_alpha.utils.dates import month_end_rebalance_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yml")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    prices = load_prices(cfg["paths"]["processed_prices"])
    predictions = pd.read_parquet(cfg["paths"]["predictions"])
    scores = predictions["p_outperform"].sort_index()

    rebalance_dates = month_end_rebalance_dates(scores.index.get_level_values("date"))
    scores = scores[scores.index.get_level_values("date").isin(rebalance_dates)]

    bt_cfg = cfg["backtest"]
    result, metrics = run_top_decile_backtest(
        prices=prices,
        scores=scores,
        horizon=cfg["labels"]["forward_horizon_days"],
        top_fraction=bt_cfg["top_fraction"],
        cost_bps=bt_cfg["transaction_cost_bps"],
        initial_capital=bt_cfg["initial_capital"],
        min_names=bt_cfg.get("min_names", 3),
    )

    backtest_path = ensure_parent_dir(cfg["paths"]["backtest"])
    metrics_path = ensure_parent_dir(cfg["paths"]["metrics"])
    result.to_csv(backtest_path)
    pd.Series(metrics, name="value").to_csv(metrics_path)
    print(f"Saved backtest to {backtest_path}")
    print(f"Saved metrics to {metrics_path}")
    print(pd.Series(metrics).round(4))


if __name__ == "__main__":
    main()
