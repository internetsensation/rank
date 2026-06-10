# prob-alpha-lab

`prob-alpha-lab` is a GitHub-style quantitative research framework for testing probabilistic machine learning models on cross-sectional equity return prediction.

The first research question is deliberately clean:

> Given today's features, what is the probability that a stock outperforms the median stock in the universe over the next month?

The first production-grade baseline is a monthly logistic-regression model that ranks stocks by predicted probability of outperformance, buys the top decile, holds for one month, and compares results against an equal-weight universe and SPY.

This is **not** a live trading bot. It is a research lab for building, testing, and stress-testing alpha signals with realistic assumptions.

---

## Core project goals

- Build a clean research pipeline from raw prices to features, labels, models, signals, backtests, and reports.
- Predict cross-sectional stock outperformance, not absolute next-day price movement.
- Estimate and use uncertainty instead of blindly ranking by raw predicted return.
- Make backtests explicit about transaction costs, turnover, rebalancing, and benchmark comparisons.
- Add Monte Carlo and stress testing before any reinforcement-learning extension.

---

## Repository structure

```text
prob-alpha-lab/
├── configs/                  # YAML experiment configuration
├── data/                     # raw/interim/processed data, ignored by git
├── notebooks/                # research notebooks
├── src/prob_alpha/           # importable Python package
├── scripts/                  # command-line pipeline entrypoints
├── reports/                  # generated figures, tables, experiment notes
├── tests/                    # pytest unit tests
└── experiments/              # saved experiment outputs
```

---

## First model: probabilistic stock outperformance

### Label

For stock `i` at date `t`, define the forward return over a future horizon, for example 21 trading days:

```text
forward_return(i, t) = adjusted_close(i, t + 21) / adjusted_close(i, t) - 1
```

Then compare each stock to the cross-sectional median forward return on the same date:

```text
label(i, t) = 1 if forward_return(i, t) > median_forward_return(t)
label(i, t) = 0 otherwise
```

The model predicts:

```text
P(stock outperforms the universe median next month | today's features)
```

### First feature set

- 12-1 momentum
- 1-month return
- 20-day realized volatility
- 60-day realized volatility
- distance from 52-week high
- average volume
- volume change
- trailing max drawdown

### First strategy

Every month:

1. Compute features for every stock.
2. Predict `P(outperform)` for every stock.
3. Rank stocks by predicted probability.
4. Buy the top 10%.
5. Hold for one month.
6. Rebalance.
7. Compare against equal-weight universe and SPY.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Optional, if you want to use Yahoo Finance data:

```bash
pip install yfinance
```

---

## Quickstart

Download data:

```bash
python scripts/download_data.py --config configs/data.yml
```

Build the supervised learning dataset:

```bash
python scripts/build_dataset.py --config configs/experiment.yml
```

Train the first probabilistic model:

```bash
python scripts/train_model.py --config configs/experiment.yml
```

Run a monthly top-decile backtest:

```bash
python scripts/run_backtest.py --config configs/experiment.yml
```

Run Monte Carlo stress tests:

```bash
python scripts/run_stress_test.py --config configs/experiment.yml
```

Run unit tests:

```bash
pytest
```

---

## Research progression

### Phase 1: baseline infrastructure

- Price loader
- Feature builder
- Label builder
- Logistic regression baseline
- Monthly top-decile backtest
- Metrics and plots

### Phase 2: probabilistic ML

- Logistic regression calibration
- Random forest classifier
- Gradient boosting classifier
- Ensemble model
- Quantile regression for return intervals
- Uncertainty-adjusted ranking

### Phase 3: better backtesting

- Transaction costs
- Turnover constraints
- Long-only top decile
- Long-short decile spread
- Sector-neutral ranking
- Volatility-scaled portfolios

### Phase 4: Monte Carlo stress testing

- Bad-luck trade sequencing
- Higher transaction costs
- Volatility spikes
- Lower signal strength
- Randomized rebalance results

### Phase 5: RL extension

RL is intentionally delayed until the supervised/probabilistic research stack is working.

Good RL targets:

- Position sizing
- Portfolio weights
- Risk exposure
- De-risking decisions

Bad first RL target:

- Predicting the next price tick

---

## Suggested interpretation of results

A model that produces a high Sharpe ratio in-sample but collapses out-of-sample is not alpha. A model that survives walk-forward testing, costs, turnover constraints, and stress tests is much more interesting.

The goal is not to make the backtest look good. The goal is to make it hard for the strategy to fool you.

---

## Educational disclaimer

This repository is for research and educational purposes only. It does not provide investment advice, portfolio advice, or trade recommendations.
