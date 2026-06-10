# RL extension

The RL module is intentionally a placeholder until the supervised/probabilistic pipeline is stable.

Recommended future setup:

- `state`: model probabilities, uncertainty estimates, volatility, current positions, market regime
- `action`: portfolio weights or risk exposure bucket
- `reward`: return - transaction costs - drawdown penalty

Do not start with tick-level price prediction. Start with position sizing and portfolio risk control.
