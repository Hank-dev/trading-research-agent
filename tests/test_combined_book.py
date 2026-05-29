import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from trading_research_agent.schemas.combined_book import CombinedBookSpec
from trading_research_agent.workflows import combined_book as cb


def make_spec(**overrides) -> CombinedBookSpec:
    data = {
        "core_assets": ["SPY"],
        "overlay_assets": ["VIXY"],
        "overlay_weight": 0.1,
        "overlay_rule": "static",
        "start_date": "2015-01-01",
        "end_date": "2024-01-01",
        "lookback_days": 50,
        "rebalance_days": 21,
    }
    data.update(overrides)
    return CombinedBookSpec(**data)


def make_panel(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2015-01-01", periods=rows, freq="D")
    spy = pd.Series(np.linspace(100, 200, rows), index=index)
    vixy = pd.Series(np.linspace(100, 10, rows), index=index)  # bleeds
    return pd.DataFrame({"SPY": spy, "VIXY": vixy})


# ---- schema ----


def test_rejects_overlapping_core_and_overlay() -> None:
    with pytest.raises(ValidationError, match="disjoint"):
        make_spec(core_assets=["SPY", "GLD"], overlay_assets=["GLD"])


def test_rejects_bad_overlay_weight() -> None:
    with pytest.raises(ValidationError, match="overlay_weight"):
        make_spec(overlay_weight=0.9)


# ---- weight logic ----


def test_static_overlay_weights_sum_to_core_plus_overlay() -> None:
    panel = make_panel()
    spec = make_spec(overlay_rule="static", overlay_weight=0.1)
    w = cb.compute_book_weights(panel, spec, include_overlay=True)
    row = w.dropna(how="all").iloc[0]
    assert row["SPY"] == pytest.approx(0.9)
    assert row["VIXY"] == pytest.approx(0.1)


def test_core_only_weights_ignore_overlay() -> None:
    panel = make_panel()
    spec = make_spec()
    w = cb.compute_book_weights(panel, spec, include_overlay=False)
    rows = w.dropna(how="all")
    assert np.allclose(rows["SPY"], 1.0)
    assert np.allclose(rows["VIXY"], 0.0)


def test_regime_overlay_only_carries_hedge_when_core_trend_down() -> None:
    rows = 300
    index = pd.date_range("2015-01-01", periods=rows, freq="D")
    up = np.linspace(100, 200, rows // 2)
    down = np.linspace(200, 90, rows - rows // 2)
    panel = pd.DataFrame(
        {"SPY": pd.Series(np.concatenate([up, down]), index=index),
         "VIXY": pd.Series(np.linspace(100, 50, rows), index=index)}
    )
    spec = make_spec(overlay_rule="regime", lookback_days=40, overlay_weight=0.2)
    w = cb.compute_book_weights(panel, spec, include_overlay=True)
    rebal = w.dropna(how="all")
    risk_on = rebal[rebal["VIXY"] == 0.0]
    risk_off = rebal[rebal["VIXY"] > 0.0]
    assert len(risk_on) > 0 and len(risk_off) > 0
    # Risk-on: full core, no hedge. Risk-off: shaved core + hedge slice.
    assert np.allclose(risk_on["SPY"], 1.0)
    assert np.allclose(risk_off["VIXY"], 0.2)
    assert np.allclose(risk_off["SPY"], 0.8)


# ---- comparison / verdict ----


def test_verdict_improves_risk_adjusted_when_sharpe_rises() -> None:
    core = {"total_return_pct": 50.0, "sharpe_ratio": 0.8, "max_drawdown_pct": -30.0, "final_equity": 15000.0}
    combined = {"total_return_pct": 45.0, "sharpe_ratio": 0.95, "max_drawdown_pct": -20.0, "final_equity": 14500.0}
    cmp = cb._compare(core, combined)
    assert cmp["verdict"] == "IMPROVES_RISK_ADJUSTED"
    assert cmp["sharpe_delta"] > 0


def test_verdict_reduces_drawdown_at_cost() -> None:
    # Sharpe slightly worse, but drawdown cut from -40 to -20 (50% shallower).
    core = {"total_return_pct": 80.0, "sharpe_ratio": 0.9, "max_drawdown_pct": -40.0, "final_equity": 18000.0}
    combined = {"total_return_pct": 55.0, "sharpe_ratio": 0.85, "max_drawdown_pct": -20.0, "final_equity": 15500.0}
    cmp = cb._compare(core, combined)
    assert cmp["verdict"] == "REDUCES_DRAWDOWN_AT_COST"
    assert cmp["drawdown_improvement_pct"] == pytest.approx(20.0)


def test_verdict_not_worth_it_when_no_benefit() -> None:
    core = {"total_return_pct": 80.0, "sharpe_ratio": 0.9, "max_drawdown_pct": -30.0, "final_equity": 18000.0}
    combined = {"total_return_pct": 40.0, "sharpe_ratio": 0.5, "max_drawdown_pct": -28.0, "final_equity": 14000.0}
    cmp = cb._compare(core, combined)
    assert cmp["verdict"] == "NOT_WORTH_IT"


def test_return_cost_is_core_minus_combined() -> None:
    core = {"total_return_pct": 80.0, "sharpe_ratio": 0.9, "max_drawdown_pct": -30.0, "final_equity": 18000.0}
    combined = {"total_return_pct": 60.0, "sharpe_ratio": 0.95, "max_drawdown_pct": -22.0, "final_equity": 16000.0}
    cmp = cb._compare(core, combined)
    assert cmp["return_cost_pct"] == pytest.approx(20.0)


# ---- lockbox orchestrator ----


def _fake_eval(verdict: str) -> dict:
    return {
        "spec_name": "x",
        "core_assets": ["SPY"],
        "overlay_assets": ["GLD"],
        "overlay_weight": 0.1,
        "overlay_rule": "static",
        "core": {"total_return_pct": 50.0, "sharpe_ratio": 0.9, "max_drawdown_pct": -30.0, "final_equity": 15000.0},
        "combined": {"total_return_pct": 45.0, "sharpe_ratio": 0.95, "max_drawdown_pct": -20.0, "final_equity": 14500.0},
        "comparison": {"verdict": verdict, "return_cost_pct": 5.0, "drawdown_improvement_pct": 10.0, "sharpe_delta": 0.05},
    }


def test_lockbox_confirmed_when_benefit_persists(monkeypatch) -> None:
    spec = make_spec(start_date="2010-01-01", end_date="2024-01-01")
    verdicts = iter(["IMPROVES_RISK_ADJUSTED", "REDUCES_DRAWDOWN_AT_COST"])  # train, lockbox
    seen_ranges: list[tuple[str, str]] = []

    def fake(s, panel):
        seen_ranges.append((s.start_date, s.end_date))
        return _fake_eval(next(verdicts))

    monkeypatch.setattr(cb, "run_combined_book_eval", fake)
    panel = make_panel()

    out = cb.run_combined_book_with_lockbox(spec, lockbox_pct=0.2, panel=panel)

    assert out["confirmed"] is True
    # Train ran on the early window, lockbox on the held-out tail.
    assert seen_ranges[0][0] == "2010-01-01"
    assert seen_ranges[0][1] < "2024-01-01"
    assert seen_ranges[1][1] == "2024-01-01"
    assert seen_ranges[1][0] > seen_ranges[0][1]


def test_lockbox_not_confirmed_when_benefit_vanishes(monkeypatch) -> None:
    spec = make_spec(start_date="2010-01-01", end_date="2024-01-01")
    verdicts = iter(["IMPROVES_RISK_ADJUSTED", "NOT_WORTH_IT"])

    monkeypatch.setattr(cb, "run_combined_book_eval", lambda s, panel: _fake_eval(next(verdicts)))

    out = cb.run_combined_book_with_lockbox(spec, lockbox_pct=0.2, panel=make_panel())
    assert out["confirmed"] is False


def test_lockbox_handles_unrunnable_held_out(monkeypatch) -> None:
    spec = make_spec(start_date="2010-01-01", end_date="2024-01-01")
    calls = {"n": 0}

    def fake(s, panel):
        calls["n"] += 1
        if calls["n"] == 2:  # the lockbox eval
            raise ValueError("segment too short")
        return _fake_eval("IMPROVES_RISK_ADJUSTED")

    monkeypatch.setattr(cb, "run_combined_book_eval", fake)
    out = cb.run_combined_book_with_lockbox(spec, lockbox_pct=0.2, panel=make_panel())
    assert out["lockbox"] is None
    assert out["confirmed"] is False


def test_lockbox_pct_zero_returns_full_only(monkeypatch) -> None:
    spec = make_spec()
    monkeypatch.setattr(cb, "run_combined_book_eval", lambda s, panel=None: _fake_eval("IMPROVES_RISK_ADJUSTED"))
    out = cb.run_combined_book_with_lockbox(spec, lockbox_pct=0.0, panel=make_panel())
    assert "full" in out
    assert "lockbox" not in out
