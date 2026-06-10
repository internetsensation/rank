from __future__ import annotations

import argparse

import pandas as pd

from prob_alpha.data.load_prices import load_prices
from prob_alpha.features.build_features import build_features
from prob_alpha.labels.outperform import outperform_median_label
from prob_alpha.utils.config import ensure_parent_dir, load_experiment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yml")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    paths = cfg["paths"]
    prices = load_prices(paths["processed_prices"])
    features = build_features(prices, cfg.get("features", {}))
    labels = outperform_median_label(prices, horizon=cfg["labels"]["forward_horizon_days"])
    dataset = features.join(labels, how="inner").dropna()

    out_path = ensure_parent_dir(paths["dataset"])
    dataset.to_parquet(out_path)
    print(f"Saved dataset to {out_path}: {dataset.shape[0]:,} rows, {dataset.shape[1]:,} columns")


if __name__ == "__main__":
    main()
