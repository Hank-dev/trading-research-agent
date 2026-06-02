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


_VOLATILITY_SCALING_WINDOW = 60


def compute_target_weights(
    close: pd.DataFrame,
    spec: PortfolioSpec,
    aux: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the (dates x assets) target-weight matrix for a PortfolioSpec.

    `aux` is an optional non-price signal panel (used by fx_carry: a
    rate-differential matrix). When provided it is reindexed onto `close.index`
    so callers (incl. walk-forward windows) can pass the full panel and stay
    positionally aligned with `close`.
    """
    if close.shape[1] < 2:
        raise ValueError("Portfolio weights require at least 2 asset columns")
    if aux is not None:
        aux = aux.reindex(close.index)

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    rebalance_rows = _rebalance_row_indices(len(close), spec)

    for i in rebalance_rows:
        weights.iloc[i] = _target_row(close, i, spec, aux).to_numpy()

    # Decide on bar i, execute on bar i+1. NaN rows remain NaN (no order).
    return weights.shift(1)


def _rebalance_row_indices(n_rows: int, spec: PortfolioSpec) -> range:
    # Start once enough history exists for the lookback window.
    return range(spec.lookback_days, n_rows, spec.rebalance_days)


def _target_row(
    close: pd.DataFrame, i: int, spec: PortfolioSpec, aux: pd.DataFrame | None = None
) -> pd.Series:
    if spec.portfolio_family == PortfolioFamily.EQUAL_WEIGHT_TREND:
        return _equal_weight_trend_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.TIME_SERIES_MOMENTUM:
        return _time_series_momentum_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.VOLATILITY_SCALED_MOMENTUM:
        return _volatility_scaled_momentum_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.CRISIS_HEDGE:
        return _crisis_hedge_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.CROSS_SECTIONAL_REVERSAL:
        return _reversal_row(close, i, spec)
    if spec.portfolio_family == PortfolioFamily.FX_CARRY:
        return _fx_carry_row(close, i, spec, aux)
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


def _reversal_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    # Long-horizon cross-sectional reversal. Score each asset by its return over
    # the window [i - lookback_days, i - skip_recent_days] — i.e. excluding the
    # most recent `skip_recent_days` bars — then hold the bottom top_k (most
    # beaten-down) equal-weight. The skip-recent gap is what separates this from
    # inverted momentum: without it, the score is dominated by the same recent
    # window momentum trades on.
    past_close = close.iloc[i - spec.lookback_days]
    recent_close = close.iloc[i - spec.skip_recent_days]
    reversal_score = (recent_close / past_close) - 1.0

    ranked = reversal_score.dropna().sort_values(ascending=True)
    selected = list(ranked.head(spec.top_k).index)
    target = pd.Series(0.0, index=close.columns)
    if selected:
        # Divide by top_k (not len(selected)) so any shortfall stays in cash,
        # matching the _momentum_row convention.
        target[selected] = 1.0 / spec.top_k
    return target


def _fx_carry_row(
    close: pd.DataFrame, i: int, spec: PortfolioSpec, aux: pd.DataFrame | None
) -> pd.Series:
    # FX carry. Score each asset by its average rate differential vs USD over the
    # last `lookback_days` (smoothing the monthly rate print to cut noise), then
    # hold the top_k highest-carry assets equal-weight. The carry differentials
    # arrive in `aux`, already publication-lagged upstream, so there is no
    # look-ahead here beyond the shared .shift(1).
    if aux is None:
        raise ValueError("fx_carry requires a carry (rate-differential) panel via aux=")
    window = aux.iloc[max(0, i - spec.lookback_days + 1) : i + 1]
    carry = window.mean()

    ranked = carry.dropna().sort_values(ascending=False)
    selected = list(ranked.head(spec.top_k).index)
    target = pd.Series(0.0, index=close.columns)
    if selected:
        # Divide by top_k (not len(selected)) so any shortfall stays in cash,
        # matching the _momentum_row convention.
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


def _volatility_scaled_momentum_row(close: pd.DataFrame, i: int, spec: PortfolioSpec) -> pd.Series:
    # Managed-futures-style absolute momentum with inverse-volatility sizing.
    # Eligible assets have positive trailing return; weights among eligible names
    # are proportional to 1 / recent realized volatility and sum to 1.0.
    lookback_close = close.iloc[i - spec.lookback_days]
    current_close = close.iloc[i]
    trailing_return = (current_close / lookback_close) - 1.0
    eligible = trailing_return > 0.0

    target = pd.Series(0.0, index=close.columns)
    if not eligible.any():
        return target

    vol_window = min(_VOLATILITY_SCALING_WINDOW, spec.lookback_days)
    recent_returns = close.pct_change().iloc[i - vol_window + 1 : i + 1]
    realized_vol = recent_returns.std(ddof=0)
    inverse_vol = (1.0 / realized_vol.where(realized_vol > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    scaled = inverse_vol[eligible].dropna()
    if scaled.empty or scaled.sum() <= 0.0:
        return target

    target[scaled.index] = scaled / scaled.sum()
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
