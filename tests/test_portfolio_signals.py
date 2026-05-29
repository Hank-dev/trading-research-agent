import numpy as np
import pandas as pd
import pytest

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.portfolio_signals import (
    compute_target_weights,
    equal_weight_benchmark_return_pct,
)


def make_spec(**overrides) -> PortfolioSpec:
    data = {
        "name": "Mom",
        "assets": ["AAA", "BBB", "CCC"],
        "portfolio_family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        "start_date": "2020-01-01",
        "end_date": "2023-01-01",
        "lookback_days": 20,
        "top_k": 1,
        "rebalance_days": 21,
        "hypothesis": "Relative strength persists across the universe.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


def make_panel(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    # AAA strong uptrend, BBB flat, CCC downtrend — momentum should pick AAA.
    aaa = pd.Series(np.linspace(100, 400, rows), index=index)
    bbb = pd.Series(np.full(rows, 100.0), index=index)
    ccc = pd.Series(np.linspace(100, 40, rows), index=index)
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc})


def test_weights_shape_matches_panel() -> None:
    panel = make_panel()
    weights = compute_target_weights(panel, make_spec())
    assert weights.shape == panel.shape
    assert list(weights.columns) == list(panel.columns)


def test_no_lookahead_first_lookback_rows_have_no_orders() -> None:
    panel = make_panel()
    spec = make_spec(lookback_days=20)
    weights = compute_target_weights(panel, spec)
    # The first rebalance is at row index 20, shifted to row 21, so rows 0..20
    # must be all-NaN (no orders possible before the lookback completes).
    assert weights.iloc[:21].isna().all().all()


def test_cross_sectional_momentum_picks_strongest_asset() -> None:
    panel = make_panel()
    spec = make_spec(top_k=1)
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    # Every rebalance should allocate 100% to AAA (strongest uptrend), 0 elsewhere.
    assert (rebalance_rows["AAA"] == 1.0).all()
    assert (rebalance_rows["BBB"] == 0.0).all()
    assert (rebalance_rows["CCC"] == 0.0).all()


def test_top_k_two_splits_equally() -> None:
    panel = make_panel()
    spec = make_spec(top_k=2)
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    # AAA and BBB are the top 2 (CCC is the downtrend); each gets 1/2.
    assert np.allclose(rebalance_rows["AAA"], 0.5)
    assert np.allclose(rebalance_rows["BBB"], 0.5)
    assert np.allclose(rebalance_rows["CCC"], 0.0)


def test_dual_momentum_goes_to_cash_when_all_negative() -> None:
    rows = 300
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    # Everything trends DOWN — dual momentum's absolute filter should hold cash.
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 50, rows), index=index),
            "BBB": pd.Series(np.linspace(100, 60, rows), index=index),
        }
    )
    spec = make_spec(
        assets=["AAA", "BBB"],
        portfolio_family=PortfolioFamily.DUAL_MOMENTUM,
        top_k=1,
    )
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    # Every rebalance row is fully in cash (all target weights zero).
    assert (rebalance_rows.sum(axis=1) == 0.0).all()


def test_dual_momentum_invests_only_in_positive_momentum() -> None:
    rows = 300
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    # AAA up, BBB down. Dual momentum holds AAA only; CCC-like down stays cash.
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 300, rows), index=index),
            "BBB": pd.Series(np.linspace(100, 50, rows), index=index),
        }
    )
    spec = make_spec(
        assets=["AAA", "BBB"],
        portfolio_family=PortfolioFamily.DUAL_MOMENTUM,
        top_k=1,
    )
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    assert (rebalance_rows["AAA"] == 1.0).all()
    assert (rebalance_rows["BBB"] == 0.0).all()


def test_equal_weight_trend_holds_only_assets_above_sma() -> None:
    rows = 300
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    # AAA uptrend (above its SMA), BBB downtrend (below its SMA).
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 400, rows), index=index),
            "BBB": pd.Series(np.linspace(400, 100, rows), index=index),
        }
    )
    spec = make_spec(
        assets=["AAA", "BBB"],
        portfolio_family=PortfolioFamily.EQUAL_WEIGHT_TREND,
        lookback_days=50,
    )
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    # AAA above SMA -> held at 1/N (=0.5); BBB below SMA -> 0.
    assert np.allclose(rebalance_rows["AAA"], 0.5)
    assert np.allclose(rebalance_rows["BBB"], 0.0)


def test_rebalance_frequency_controls_order_count() -> None:
    panel = make_panel(rows=300)
    frequent = compute_target_weights(panel, make_spec(rebalance_days=5))
    rare = compute_target_weights(panel, make_spec(rebalance_days=60))
    assert frequent.notna().any(axis=1).sum() > rare.notna().any(axis=1).sum()


def test_equal_weight_benchmark_return() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    # AAA doubles, BBB halves -> equal-weight ends at mean(2.0, 0.5)=1.25 -> +25%.
    panel = pd.DataFrame(
        {"AAA": [100.0, 150.0, 200.0], "BBB": [100.0, 75.0, 50.0]}, index=index
    )
    assert equal_weight_benchmark_return_pct(panel) == pytest.approx(25.0)


def test_compute_weights_rejects_single_column() -> None:
    panel = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        compute_target_weights(panel, make_spec())


def test_time_series_momentum_holds_only_positive_trend_assets() -> None:
    panel = make_panel()  # AAA up, BBB flat, CCC down
    spec = make_spec(portfolio_family=PortfolioFamily.TIME_SERIES_MOMENTUM)
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    # AAA trailing return > 0 -> held at 1/N (=1/3). BBB flat and CCC down -> cash.
    assert np.allclose(rebalance_rows["AAA"], 1.0 / 3.0)
    assert np.allclose(rebalance_rows["BBB"], 0.0)
    assert np.allclose(rebalance_rows["CCC"], 0.0)


def test_time_series_momentum_goes_all_cash_when_all_falling() -> None:
    rows = 300
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    panel = pd.DataFrame(
        {
            "AAA": pd.Series(np.linspace(100, 50, rows), index=index),
            "BBB": pd.Series(np.linspace(100, 70, rows), index=index),
        }
    )
    spec = make_spec(
        assets=["AAA", "BBB"], portfolio_family=PortfolioFamily.TIME_SERIES_MOMENTUM
    )
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")
    assert (rebalance_rows.sum(axis=1) == 0.0).all()


def test_crisis_hedge_holds_core_in_calm_hedge_in_stress() -> None:
    rows = 300
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    # Core rises for the first half (above SMA), then crashes below SMA.
    up = np.linspace(100, 200, rows // 2)
    down = np.linspace(200, 80, rows - rows // 2)
    core = pd.Series(np.concatenate([up, down]), index=index)
    # Hedge spikes during the crash (mimics a vol ETF).
    hedge = pd.Series(
        np.concatenate([np.full(rows // 2, 20.0), np.linspace(20, 60, rows - rows // 2)]),
        index=index,
    )
    panel = pd.DataFrame({"SPY": core, "VIXY": hedge})
    spec = make_spec(
        assets=["SPY", "VIXY"],
        portfolio_family=PortfolioFamily.CRISIS_HEDGE,
        lookback_days=50,
        hedge_weight=0.2,
    )
    weights = compute_target_weights(panel, spec)
    rebalance_rows = weights.dropna(how="all")

    # Some rebalances are risk-on (core full), some risk-off (hedge slice, no core).
    risk_on = rebalance_rows[(rebalance_rows["SPY"] == 1.0)]
    risk_off = rebalance_rows[(rebalance_rows["SPY"] == 0.0)]
    assert len(risk_on) > 0
    assert len(risk_off) > 0
    # Risk-off rows hold exactly hedge_weight in VIXY and nothing in SPY.
    assert np.allclose(risk_off["VIXY"], 0.2)
    # Risk-on rows never hold the bleeding vol ETF.
    assert np.allclose(risk_on["VIXY"], 0.0)
