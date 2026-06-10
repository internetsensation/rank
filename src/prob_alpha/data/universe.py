from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Universe:
    """Equity universe definition."""

    tickers: list[str]
    benchmark: str = "SPY"

    @property
    def all_symbols(self) -> list[str]:
        symbols = list(dict.fromkeys([*self.tickers, self.benchmark]))
        return symbols


def from_config(config: dict) -> Universe:
    universe = config.get("universe", {})
    return Universe(
        tickers=list(universe.get("tickers", [])),
        benchmark=universe.get("benchmark", "SPY"),
    )
