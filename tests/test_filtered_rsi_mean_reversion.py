"""Tests for the filtered_rsi_mean_reversion family.

Plain RSI mean reversion buys whenever RSI is oversold. The filtered version adds
a calm-volatility gate: only buy the dip when ATR is BELOW its own moving average
(volatility not spiking). The filter must only REMOVE entries, never add them.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trading_research_agent.backtesting.backends.vectorbt_backend import VectorbtBackend
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.workflows import parameter_sweep


def make_spec(family=StrategyFamily.FILTERED_RSI_MEAN_REVERSION, **ov) -> StrategySpec:
    data = {
        "name": "FMR",
        "asset": "DBC",
        "strategy_family": family,
        "start_date": "2010-01-01",
        "end_date": "2020-01-01",
        "rsi_window": 14,
        "oversold_threshold": 30.0,
        "exit_threshold": 50.0,
        "atr_window": 14,
        "atr_ma_window": 20,
        "hypothesis": "Mean reversion works better when volatility is calm.",
    }
    data.update(ov)
    return StrategySpec(**data)


def _ohlc() -> pd.DataFrame:
    # Phase A (0-29): flat ~100, moderate range -> seeds the ATR average.
    # Phase B (30-79): gentle decline 100->85, SHRINKING range -> ATR below its MA
    #                  (calm); the decline drives RSI < 30 -> filtered should fire.
    # Phase C (80-114): decline 85->70 with a HUGE constant range -> ATR spikes above
    #                  its MA; RSI still < 30 -> plain fires here, filtered must NOT.
    closes, ranges = [], []
    closes += [100.0] * 30
    ranges += [1.0] * 30
    closes += list(np.linspace(100, 85, 50))
    ranges += list(np.linspace(0.8, 0.1, 50))
    closes += list(np.linspace(85, 70, 35))
    ranges += [6.0] * 35
    idx = pd.date_range("2015-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=idx)
    rng = pd.Series(ranges, index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + rng / 2,
            "Low": close - rng / 2,
            "Close": close,
            "Volume": pd.Series(np.full(len(closes), 1_000.0), index=idx),
        }
    )


def test_calm_vol_filter_gates_entries() -> None:
    data = _ohlc()
    backend = VectorbtBackend()
    entries_f, _ = backend._signals(make_spec(), data)
    entries_plain, _ = backend._signals(
        make_spec(family=StrategyFamily.RSI_MEAN_REVERSION), data
    )

    # Calm + oversold episode (phase B) fires for the filtered family.
    assert entries_f.iloc[30:80].any()
    # Vol-spike episode (phase C) is gated out for filtered...
    assert not entries_f.iloc[80:115].any()
    # ...even though plain RSI WOULD have fired there (proving the filter removed it).
    assert entries_plain.iloc[80:115].any()


def test_filtered_entries_are_subset_of_plain() -> None:
    data = _ohlc()
    backend = VectorbtBackend()
    entries_f, _ = backend._signals(make_spec(), data)
    entries_plain, _ = backend._signals(
        make_spec(family=StrategyFamily.RSI_MEAN_REVERSION), data
    )
    # Every filtered entry is also a plain entry; the filter only removes.
    assert not (entries_f & ~entries_plain).any()
    # And it actually removed something (otherwise the filter is a no-op here).
    assert (entries_plain & ~entries_f).any()


def test_filtered_mr_requires_atr_params() -> None:
    with pytest.raises(ValidationError, match="atr"):
        make_spec(atr_window=None)


def test_filtered_mr_requires_rsi_params() -> None:
    with pytest.raises(ValidationError, match="rsi|oversold|exit"):
        make_spec(oversold_threshold=None)


def test_sweep_exposes_filtered_mr() -> None:
    assert StrategyFamily.FILTERED_RSI_MEAN_REVERSION in parameter_sweep._FAMILY_DEFAULTS
    assert (
        "oversold_threshold"
        in parameter_sweep._SWEEPABLE[StrategyFamily.FILTERED_RSI_MEAN_REVERSION]
    )
