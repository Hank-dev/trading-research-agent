"""Tests for the cross_sectional_reversal portfolio family.

Reversal is structurally a sibling of cross_sectional_momentum, but it ranks
ascending (buys the most beaten-down assets) over a *gapped* window that
excludes the most recent `skip_recent_days`. The skip-recent gap is the
load-bearing detail that makes long-horizon reversal orthogonal to momentum
rather than merely inverted momentum.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import trading_research_agent.workflows.portfolio_batch as pb
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.portfolio_signals import compute_target_weights


def make_reversal_spec(**overrides) -> PortfolioSpec:
    data = {
        "name": "Rev",
        "assets": ["AAA", "BBB", "CCC"],
        "portfolio_family": PortfolioFamily.CROSS_SECTIONAL_REVERSAL,
        "start_date": "2010-01-01",
        "end_date": "2020-01-01",
        "lookback_days": 1008,
        "skip_recent_days": 252,
        "top_k": 1,
        "rebalance_days": 21,
        "hypothesis": "Long-horizon losers mean-revert.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


# --- signal behaviour -------------------------------------------------------


def test_reversal_picks_long_horizon_loser() -> None:
    # AAA up, BBB flat, CCC down -> reversal should hold the worst (CCC).
    rows = 300
    index = pd.date_range("2010-01-01", periods=rows, freq="D")
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 400, rows), index=index),
            "BBB": pd.Series(np.full(rows, 100.0), index=index),
            "CCC": pd.Series(np.linspace(100, 40, rows), index=index),
        }
    )
    spec = make_reversal_spec(lookback_days=20, skip_recent_days=5, top_k=1)
    weights = compute_target_weights(panel, spec)
    rebalances = weights.dropna(how="all")
    assert (rebalances["CCC"] == 1.0).all()
    assert (rebalances["AAA"] == 0.0).all()
    assert (rebalances["BBB"] == 0.0).all()


def _skip_panel() -> pd.DataFrame:
    # Single rebalance at i=40 (lookback=40), so gapped window is rows 0->30 and
    # the last 10 bars (rows 30->40) are the recent period that must be ignored.
    # AAA crashes over the gapped window but rallies hard in the recent window;
    # a gapped reversal must still pick AAA, a no-skip reversal must not.
    rows = 120
    index = pd.date_range("2010-01-01", periods=rows, freq="D")

    def seg(a, b, n):
        return list(np.linspace(a, b, n))

    aaa = seg(100, 50, 31)[:-1] + seg(50, 200, 11)  # 0..30 down, 30..40 rally
    aaa += [200.0] * (rows - len(aaa))
    ccc = seg(100, 150, 31)[:-1] + seg(150, 120, 11)  # 0..30 up, 30..40 pullback
    ccc += [120.0] * (rows - len(ccc))
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(aaa, index=index),
            "BBB": pd.Series(np.full(rows, 100.0), index=index),
            "CCC": pd.Series(ccc, index=index),
        }
    )
    return panel


def test_reversal_is_driven_by_gapped_window_not_recent_rally() -> None:
    panel = _skip_panel()
    spec = make_reversal_spec(
        lookback_days=40, skip_recent_days=10, rebalance_days=80, top_k=1
    )
    weights = compute_target_weights(panel, spec)
    rebalances = weights.dropna(how="all")
    assert len(rebalances) == 1
    # Gapped score over rows 0->30: AAA -50%, BBB 0%, CCC +50% -> reversal = AAA.
    assert rebalances.iloc[0]["AAA"] == 1.0


def test_no_skip_reversal_makes_a_different_pick() -> None:
    # With skip_recent_days=0 the recent rally is counted, so AAA is no longer
    # the worst and the pick changes. Proves the gap is load-bearing.
    panel = _skip_panel()
    spec = make_reversal_spec(
        lookback_days=40, skip_recent_days=0, rebalance_days=80, top_k=1
    )
    weights = compute_target_weights(panel, spec)
    rebalances = weights.dropna(how="all")
    # Full-window score to row 40: AAA +100%, BBB 0%, CCC +20% -> worst = BBB.
    assert rebalances.iloc[0]["BBB"] == 1.0
    assert rebalances.iloc[0]["AAA"] == 0.0


def test_reversal_has_no_lookahead() -> None:
    panel = _skip_panel()
    spec = make_reversal_spec(
        lookback_days=40, skip_recent_days=10, rebalance_days=80, top_k=1
    )
    weights = compute_target_weights(panel, spec)
    # Decision at row 40 is shifted to row 41; rows 0..40 carry no orders.
    assert weights.iloc[:41].isna().all().all()
    assert weights.iloc[41].notna().any()


# --- schema validation ------------------------------------------------------


def test_skip_recent_must_be_less_than_lookback() -> None:
    with pytest.raises(ValidationError, match="skip_recent_days"):
        make_reversal_spec(lookback_days=60, skip_recent_days=60)


def test_skip_recent_must_not_be_negative() -> None:
    with pytest.raises(ValidationError, match="skip_recent_days"):
        make_reversal_spec(lookback_days=60, skip_recent_days=-1)


def test_lookback_ceiling_raised_to_1260() -> None:
    spec = make_reversal_spec(lookback_days=1260, skip_recent_days=252)
    assert spec.lookback_days == 1260
    with pytest.raises(ValidationError, match="lookback_days"):
        make_reversal_spec(lookback_days=1261, skip_recent_days=252)


def test_reversal_enforces_top_k_bounds() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        make_reversal_spec(top_k=99)  # more than the 3-asset universe


# --- batch loader -----------------------------------------------------------


def test_batch_loader_round_trips_skip_recent(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            [
                {
                    "family": "cross_sectional_reversal",
                    "assets": ["SPY", "TLT", "GLD"],
                    "start": "2010-01-01",
                    "end": "2020-01-01",
                    "lookback_days": 1008,
                    "skip_recent": 252,
                    "top_k": 1,
                }
            ]
        )
    )
    specs = pb.load_portfolio_batch(path)
    assert specs[0].portfolio_family == PortfolioFamily.CROSS_SECTIONAL_REVERSAL
    assert specs[0].skip_recent_days == 252


def test_batch_loader_skip_recent_defaults(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            [
                {
                    "family": "cross_sectional_reversal",
                    "assets": ["SPY", "TLT", "GLD"],
                    "start": "2010-01-01",
                    "end": "2020-01-01",
                    "lookback_days": 1008,
                    "top_k": 1,
                }
            ]
        )
    )
    specs = pb.load_portfolio_batch(path)
    assert specs[0].skip_recent_days == 252
