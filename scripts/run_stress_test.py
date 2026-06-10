from __future__ import annotations

import argparse

import pandas as pd

from prob_alpha.risk.monte_carlo import bootstrap_return_paths, summarize_bootstrap_paths
from prob_alpha.risk.stress_tests import apply_cost_shock, apply_signal_decay, apply_volatility_spike
from prob_alpha.utils.config import ensure_parent_dir, load_experiment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yml")
    parser.add_argument("--paths", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    backtest = pd.read_csv(cfg["paths"]["backtest"], parse_dates=["date"], index_col="date")
    returns = backtest["net_return"]

    stressed = pd.DataFrame(
        {
            "base": returns,
            "higher_costs": apply_cost_shock(returns, 0.001),
            "signal_decay_50pct": apply_signal_decay(returns, 0.50),
            "vol_spike_150pct": apply_volatility_spike(returns, 1.50),
        }
    )
    paths = bootstrap_return_paths(returns, n_paths=args.paths)
    summary = summarize_bootstrap_paths(paths)

    stressed_path = ensure_parent_dir("reports/tables/stress_scenarios.csv")
    mc_path = ensure_parent_dir("reports/tables/monte_carlo_summary.csv")
    stressed.to_csv(stressed_path)
    summary.describe().to_csv(mc_path)
    print(f"Saved stress scenarios to {stressed_path}")
    print(f"Saved Monte Carlo summary to {mc_path}")


if __name__ == "__main__":
    main()
