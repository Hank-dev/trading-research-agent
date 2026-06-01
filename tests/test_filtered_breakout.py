import os

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.tools.indicators import atr
from trading_research_agent.workflows import parameter_sweep as ps


# ---- ATR indicator ----


def test_atr_is_nonnegative_and_tracks_range() -> None:
    n = 50
    high = pd.Series(np.linspace(10, 20, n)) + 1.0
    low = pd.Series(np.linspace(10, 20, n)) - 1.0
    close = pd.Series(np.linspace(10, 20, n))
    a = atr(high, low, close, window=14)
    valid = a.dropna()
    assert (valid >= 0).all()
    # With a constant ~2-wide bar and steady drift, ATR should be ~2-ish.
    assert 1.0 < valid.iloc[-1] < 4.0


def test_atr_warmup_is_nan() -> None:
    n = 30
    s = pd.Series(np.arange(n, dtype=float))
    a = atr(s + 1, s - 1, s, window=14)
    assert a.iloc[:13].isna().all()


# ---- schema validation ----


def make_filtered(**overrides) -> StrategySpec:
    data = {
        "name": "FiltBO",
        "asset": "SPY",
        "strategy_family": StrategyFamily.FILTERED_DONCHIAN_BREAKOUT,
        "start_date": "2010-01-01",
        "end_date": "2024-01-01",
        "entry_window": 55,
        "exit_window": 20,
        "atr_window": 14,
        "atr_ma_window": 20,
        "regime_window": 200,
        "hypothesis": "Filtered breakout avoids chop fakeouts.",
    }
    data.update(overrides)
    return StrategySpec(**data)


def test_filtered_breakout_valid() -> None:
    spec = make_filtered()
    assert spec.strategy_family == StrategyFamily.FILTERED_DONCHIAN_BREAKOUT
    assert spec.regime_window == 200


def test_filtered_breakout_requires_filter_windows() -> None:
    with pytest.raises(ValidationError, match="atr_window"):
        make_filtered(atr_window=None)


def test_filtered_breakout_bounds_regime_window() -> None:
    with pytest.raises(ValidationError, match="regime_window"):
        make_filtered(regime_window=500)


# ---- signal gating (the core new logic) ----


def _make_ohlc(close_vals: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=len(close_vals), freq="D")
    close = pd.Series(close_vals, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1000.0},
        index=idx,
    )


def test_filter_blocks_breakout_in_downtrend() -> None:
    pytest.importorskip("vectorbt")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from trading_research_agent.backtesting.backends.vectorbt_backend import VectorbtBackend

    # Steadily DOWN-trending series: any breakout is below the regime MA, so the
    # uptrend filter must block all entries.
    data = _make_ohlc(list(np.linspace(300, 100, 300)))
    spec = make_filtered(
        entry_window=20, exit_window=10, atr_window=14, atr_ma_window=20, regime_window=100
    )
    entries, exits = VectorbtBackend()._signals(spec, data)
    assert entries.sum() == 0  # downtrend regime blocks every long entry


def test_filtered_has_fewer_entries_than_plain_donchian() -> None:
    pytest.importorskip("vectorbt")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from trading_research_agent.backtesting.backends.vectorbt_backend import VectorbtBackend

    # Choppy + drifting series so plain Donchian fires repeatedly.
    rng = np.random.default_rng(0)
    vals = 100 + np.cumsum(rng.normal(0.05, 2.0, 400))
    data = _make_ohlc(list(vals))
    backend = VectorbtBackend()

    plain = make_filtered().model_copy(
        update={"strategy_family": StrategyFamily.DONCHIAN_BREAKOUT, "entry_window": 20, "exit_window": 10}
    )
    filtered = make_filtered(entry_window=20, exit_window=10, regime_window=100)

    plain_entries, _ = backend._signals(plain, data)
    filt_entries, _ = backend._signals(filtered, data)
    # Filters can only remove entries, never add them.
    assert filt_entries.sum() <= plain_entries.sum()


# ---- sweep registration ----


def test_sweep_recognizes_filtered_family() -> None:
    assert StrategyFamily.FILTERED_DONCHIAN_BREAKOUT in ps._FAMILY_DEFAULTS
    assert "entry_window" in ps._SWEEPABLE[StrategyFamily.FILTERED_DONCHIAN_BREAKOUT]
    assert "regime_window" in ps._SWEEPABLE[StrategyFamily.FILTERED_DONCHIAN_BREAKOUT]
