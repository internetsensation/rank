from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prob_alpha.rl.rewards import risk_adjusted_reward


@dataclass
class PortfolioSizingEnv:
    """Minimal placeholder environment for future RL portfolio sizing experiments.

    This is not intended to be a full Gymnasium environment yet. It is deliberately small so the
    research stack stays focused on supervised probabilistic modeling first.
    """

    returns: pd.Series
    transaction_cost: float = 0.0005
    drawdown_penalty: float = 1.0

    def reset(self):
        self.t = 0
        self.equity = 1.0
        self.peak = 1.0
        return self._state()

    def step(self, action_weight: float):
        action_weight = float(np.clip(action_weight, 0.0, 1.0))
        r = float(self.returns.iloc[self.t]) * action_weight
        cost = self.transaction_cost * abs(action_weight)
        self.equity *= 1 + r - cost
        self.peak = max(self.peak, self.equity)
        drawdown = self.equity / self.peak - 1.0
        reward = risk_adjusted_reward(r, cost, drawdown, self.drawdown_penalty)
        self.t += 1
        done = self.t >= len(self.returns)
        return self._state(), reward, done, {"equity": self.equity, "drawdown": drawdown}

    def _state(self):
        if self.t >= len(self.returns):
            return np.array([0.0, self.equity, self.equity / self.peak - 1.0])
        return np.array([float(self.returns.iloc[self.t]), self.equity, self.equity / self.peak - 1.0])
