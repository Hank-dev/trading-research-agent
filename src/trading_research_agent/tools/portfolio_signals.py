"""Target-weight construction for multi-asset portfolio strategies.

Kept as pure pandas (no vectorbt) so the look-ahead-sensitive logic can be unit
tested in isolation. The output is a (dates x assets) target-weight matrix to be
fed to the portfolio backend:

- A row of NaN means "no rebalance today — hold whatever you have."
- A row of explicit weights (summing to <= 1, the remainder is cash) means
  "rebalance to these target percentages today."

All decisions are made from data available up to and including a rebalance day,
then shifted forward one day so orders execute on the *next* bar. This is the
single most important guard against accidentally trading on information you
would not have had yet.
"""

import numpy as np
import pandas as pd

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.indicators import sma


def compute_target_weights(close: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    """Build the (dates x assets) target-weight matrix for a PortfolioSpec."""
    if close.shape[1] < 2:
        raise ValueError("Portfolio weights require at least 2 asset columns")

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    rebalance_rows = _rebalance_row_indices(len(close), spec)

    for i in rebalance_rows:
        weights.iloc[i] = _target_row(close, i, spec).to_numpy()

    # Decide on bar i, execute on bar i+1. NaN rows remain NaN (no order).
    return weights.shift(1)


def _rebalance_row_indices(n_rows: int, spec: PortfolioSpec) -> range:
    # Start once enough history exists for the lookback window.
    return range(spec.lookback_days, n_rows, spec.rebalance_days)


def _target_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    if spec.portfolio_family == PortfolioFamily.EQUAL_WEIGHT_TREND:
        return _equal_weight_trend_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.TIME_SERIES_MOMENTUM:
        return _time_series_momentum_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.CRISIS_HEDGE:
        return _crisis_hedge_row(close, i, spec)
    return _momentum_row(close, i, spec)


def _momentum_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    lookback_close = close.iloc[i - spec.lookback_days]
    current_close = close.iloc[i]
    momentum = (current_close / lookback_close) - 1.0

    ranked = momentum.dropna().sort_values(ascending=False)
    if spec.portfolio_family == PortfolioFamily.DUAL_MOMENTUM:
        # Absolute filter: only hold assets with positive trailing return.
        ranked = ranked[ranked > 0.0]

    selected = list(ranked.head(spec.top_k).index)
    target = pd.Series(0.0, index=close.columns)
    if selected:
        # Divide by top_k (not len(selected)) so that when the dual-momentum
        # filter leaves fewer than top_k qualifiers, the remainder stays in cash.
        target[selected] = 1.0 / spec.top_k
    return target


def _equal_weight_trend_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    # Per-asset trend filter: hold an asset only while above its own SMA.
    window = close.iloc[: i + 1]
    sma_now = window.apply(lambda col: sma(col, spec.lookback_days).iloc[-1])
    current_close = close.iloc[i]
    above = (current_close > sma_now) & sma_now.notna()
    target = pd.Series(0.0, index=close.columns)
    if above.any():
        target[above] = 1.0 / len(close.columns)
    return target


def _time_series_momentum_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    # Per-asset absolute momentum: hold an equal-weight slice (1/N of the whole
    # universe) only while the asset's own trailing return is positive. Assets
    # that switch off leave their slice in cash, so a broad downturn de-risks the
    # book automatically.
    lookback_close = close.iloc[i - spec.lookback_days]
    current_close = close.iloc[i]
    trailing_return = (current_close / lookback_close) - 1.0
    on = trailing_return > 0.0
    target = pd.Series(0.0, index=close.columns)
    if on.any():
        target[on] = 1.0 / len(close.columns)
    return target


def _crisis_hedge_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    # Two-asset overlay [core, volatility_hedge]. Hold 100% core while it is above
    # its own SMA; when it breaks below, exit core to cash and hold a capped slice
    # of the volatility hedge. The hedge is NEVER held in calm regimes — that is
    # what keeps a bleeding vol ETF from destroying the strategy.
    core, hedge = close.columns[0], close.columns[1]
    core_window = close[core].iloc[: i + 1]
    sma_now = sma(core_window, spec.lookback_days).iloc[-1]
    core_price = close[core].iloc[i]

    target = pd.Series(0.0, index=close.columns)
    if pd.notna(sma_now) and core_price > sma_now:
        target[core] = 1.0
    else:
        target[hedge] = float(spec.hedge_weight or 0.0)
    return target


def equal_weight_benchmark_return_pct(close: pd.DataFrame) -> float:
    """Total return of an equal-weight buy-and-hold of the universe, in percent."""
    normalized = close.div(close.iloc[0])
    equity = normalized.mean(axis=1)
    return float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
