from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.Series, title: str, output_path: str | Path | None = None):
    """Plot an equity curve and optionally save it."""
    ax = equity.plot(figsize=(10, 5), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig = ax.get_figure()
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig
