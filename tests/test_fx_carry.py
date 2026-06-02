"""Tests for the fx_carry portfolio family.

Two units under test, both pure (no network):
- `_carry_panel_from_rates`: turns monthly interest-rate series into a daily
  rate-differential (carry) panel, applying a publication lag so a month's rate
  is invisible until `lag_days` after its stamp (look-ahead guard).
- `_fx_carry_row` (via `compute_target_weights` with an `aux` carry panel): longs
  the top_k highest-carry assets.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.data_loader import _carry_panel_from_rates
from trading_research_agent.tools.portfolio_signals import compute_target_weights


# --- pure carry-differential transform + publication lag --------------------


def test_carry_differential_and_publication_lag() -> None:
    # Monthly rates stamped at month-start. AAA jumps 3->5 in Feb; USD flat at 1.
    rates = {
        "AAA": pd.Series(
            [3.0, 5.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"])
        ),
        "BBB": pd.Series(
            [0.0, 0.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"])
        ),
    }
    usd = pd.Series([1.0, 1.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    idx = pd.date_range("2020-01-01", "2020-04-30", freq="D")

    carry = _carry_panel_from_rates(rates, usd, idx, lag_days=40)

    assert list(carry.columns) == ["AAA", "BBB"]
    # Before any rate is available (+40d from 2020-01-01 = 2020-02-10): NaN.
    assert pd.isna(carry.loc["2020-01-15", "AAA"])
    # 2020-02-15: Jan rate is available (since 02-10) but Feb rate (stamp 02-01,
    # available 03-12) is NOT yet -> carry uses Jan: 3 - 1 = 2.0, NOT 5 - 1 = 4.0.
    assert carry.loc["2020-02-15", "AAA"] == 2.0
    # 2020-03-20: Feb rate now available -> 5 - 1 = 4.0.
    assert carry.loc["2020-03-20", "AAA"] == 4.0
    assert carry.loc["2020-03-20", "BBB"] == -1.0


# --- signal behaviour -------------------------------------------------------


def _close_panel(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    # Prices are irrelevant to the carry signal; make them distinct constants.
    return pd.DataFrame(
        {
            "AAA": pd.Series(np.full(rows, 100.0), index=idx),
            "BBB": pd.Series(np.full(rows, 100.0), index=idx),
            "CCC": pd.Series(np.full(rows, 100.0), index=idx),
        }
    )


def _carry_panel(close: pd.DataFrame, a: float, b: float, c: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"AAA": a, "BBB": b, "CCC": c}, index=close.index, dtype=float
    )


def make_carry_spec(**overrides) -> PortfolioSpec:
    data = {
        "name": "Carry",
        "assets": ["AAA", "BBB", "CCC"],
        "portfolio_family": PortfolioFamily.FX_CARRY,
        "start_date": "2020-01-01",
        "end_date": "2021-01-01",
        "lookback_days": 21,
        "top_k": 1,
        "rebalance_days": 21,
        "hypothesis": "High-rate currencies earn carry.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


def test_fx_carry_longs_highest_carry_asset() -> None:
    close = _close_panel()
    aux = _carry_panel(close, a=5.0, b=2.0, c=1.0)  # AAA highest carry
    weights = compute_target_weights(close, make_carry_spec(top_k=1), aux=aux)
    rebalances = weights.dropna(how="all")
    assert (rebalances["AAA"] == 1.0).all()
    assert (rebalances["BBB"] == 0.0).all()
    assert (rebalances["CCC"] == 0.0).all()


def test_fx_carry_top_k_two_splits_across_top_two() -> None:
    close = _close_panel()
    aux = _carry_panel(close, a=5.0, b=2.0, c=1.0)  # AAA, BBB are top-2
    weights = compute_target_weights(close, make_carry_spec(top_k=2), aux=aux)
    rebalances = weights.dropna(how="all")
    assert (rebalances["AAA"] == 0.5).all()
    assert (rebalances["BBB"] == 0.5).all()
    assert (rebalances["CCC"] == 0.0).all()


def test_fx_carry_requires_aux_panel() -> None:
    close = _close_panel()
    with pytest.raises(ValueError, match="carry"):
        compute_target_weights(close, make_carry_spec(), aux=None)


def test_aux_is_ignored_by_non_carry_families() -> None:
    # Passing aux to a momentum spec must not break it (aux is carry-only).
    rows = 300
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 400, rows), index=idx),  # strongest
            "BBB": pd.Series(np.full(rows, 100.0), index=idx),
            "CCC": pd.Series(np.linspace(100, 40, rows), index=idx),
        }
    )
    aux = _carry_panel(close, a=1.0, b=9.0, c=9.0)  # would favour BBB/CCC if used
    spec = make_carry_spec(portfolio_family=PortfolioFamily.CROSS_SECTIONAL_MOMENTUM)
    weights = compute_target_weights(close, spec, aux=aux)
    rebalances = weights.dropna(how="all")
    assert (rebalances["AAA"] == 1.0).all()  # momentum still picks AAA


# --- schema -----------------------------------------------------------------


def test_fx_carry_enforces_top_k_bounds() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        make_carry_spec(top_k=99)
