from __future__ import annotations


def risk_adjusted_reward(
    portfolio_return: float,
    transaction_cost: float = 0.0,
    drawdown: float = 0.0,
    drawdown_penalty: float = 1.0,
) -> float:
    """Simple reward for future RL experiments."""
    return portfolio_return - transaction_cost - drawdown_penalty * abs(min(drawdown, 0.0))
