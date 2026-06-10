from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from prob_alpha.models.logistic_model import make_logistic_pipeline, predict_outperformance_probability
from prob_alpha.utils.config import ensure_parent_dir, load_experiment_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yml")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    dataset = pd.read_parquet(cfg["paths"]["dataset"])
    target = "outperform_median"
    exclude = {"forward_return", "cross_sectional_median_return", target}
    feature_cols = [c for c in dataset.columns if c not in exclude]

    dates = dataset.index.get_level_values("date")
    train = dataset[(dates >= cfg["splits"]["train_start"]) & (dates <= cfg["splits"]["train_end"])]
    test = dataset[dates >= cfg["splits"]["test_start"]]
    if cfg["splits"].get("test_end"):
        test = test[test.index.get_level_values("date") <= cfg["splits"]["test_end"]]

    params = cfg.get("model", {}).get("params", {})
    model = make_logistic_pipeline(**params)
    model.fit(train[feature_cols], train[target])

    p = predict_outperformance_probability(model, test[feature_cols])
    pred = test[["forward_return", target]].join(p)
    out_path = ensure_parent_dir(cfg["paths"]["predictions"])
    pred.to_parquet(out_path)

    y_true = test[target]
    print(f"Saved predictions to {out_path}")
    print(f"Test rows: {len(test):,}")
    print(f"Log loss: {log_loss(y_true, p):.4f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y_true, p):.4f}")
    except ValueError:
        print("ROC AUC unavailable because test labels contain one class.")


if __name__ == "__main__":
    main()
