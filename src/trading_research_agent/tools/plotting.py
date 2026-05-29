from datetime import UTC, datetime
import os
from pathlib import Path
import re

import pandas as pd

matplotlib_cache = Path("/tmp/trading_research_agent_matplotlib")
matplotlib_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def save_equity_curve(equity_curve: pd.Series, strategy_name: str) -> str:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", strategy_name).strip("_").lower()
    path = output_dir / f"{timestamp}_{safe_name}_equity_curve.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve.plot(ax=ax)
    ax.set_title(f"{strategy_name} Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
