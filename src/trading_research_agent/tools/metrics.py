import math
from typing import Any

import pandas as pd

from trading_research_agent.schemas.backtest import BacktestMetrics


def calculate_buy_and_hold_return(data: pd.DataFrame) -> float:
    closes = data["Close"].dropna()
    if len(closes) < 2:
        raise ValueError("At least two close prices are required for buy-and-hold return")
    return float((closes.iloc[-1] / closes.iloc[0] - 1) * 100)


def extract_backtesting_py_metrics(
    stats: pd.Series, buy_and_hold_return_pct: float
) -> BacktestMetrics:
    total_return_pct = _required_float(stats, "Return [%]")
    return BacktestMetrics(
        total_return_pct=total_return_pct,
        buy_and_hold_return_pct=buy_and_hold_return_pct,
        sharpe_ratio=_optional_float(stats.get("Sharpe Ratio")),
        max_drawdown_pct=_required_float(stats, "Max. Drawdown [%]"),
        num_trades=int(_required_float(stats, "# Trades")),
        win_rate_pct=_optional_float(stats.get("Win Rate [%]")),
        exposure_time_pct=_optional_float(stats.get("Exposure Time [%]")),
        final_equity=_required_float(stats, "Equity Final [$]"),
        beats_benchmark=total_return_pct > buy_and_hold_return_pct,
    )


def _required_float(stats: pd.Series, key: str) -> float:
    value = stats.get(key)
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"Required backtest metric unavailable: {key}")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed
