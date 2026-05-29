from pathlib import Path

import pandas as pd
import pytest

from trading_research_agent.workflows import paper_trading as pt


def make_position(**overrides) -> dict:
    data = {
        "id": "abc123",
        "created_at": "2026-05-29T00:00:00Z",
        "inception_date": "2025-01-01",
        "strategy_family": "cross_sectional_momentum",
        "params": {"assets": ["SPY", "TLT", "GLD"], "lookback_days": 126, "top_k": 2, "rebalance_days": 21},
        "initial_cash": 10_000,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "expectation": {
            "annualized_return_pct": 12.0,
            "backtest_total_return_pct": 180.0,
            "backtest_span_days": 5400,
            "backtest_sharpe": 1.0,
            "backtest_max_drawdown_pct": -16.0,
        },
        "status": "open",
    }
    data.update(overrides)
    return data


# ---- storage ----


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "paper.jsonl"
    pt.append_paper_position(make_position(id="a"), path=path)
    pt.append_paper_position(make_position(id="b"), path=path)
    loaded = pt.load_paper_positions(path=path)
    assert [p["id"] for p in loaded] == ["a", "b"]


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert pt.load_paper_positions(path=tmp_path / "nope.jsonl") == []


# ---- annualize ----


def test_annualize_one_year_is_identity() -> None:
    assert pt.annualize(10.0, 365) == pytest.approx(10.0, abs=0.2)


def test_annualize_scales_up_short_periods() -> None:
    # 10% over half a year annualizes to ~21%.
    assert pt.annualize(10.0, 182) == pytest.approx(21.0, abs=1.0)


def test_annualize_handles_total_loss() -> None:
    # Should not raise on <= -100% returns.
    assert pt.annualize(-100.0, 365) <= 0.0


# ---- read logic ----


def test_read_flags_drawdown_breach() -> None:
    read, _ = pt._read(
        forward_trading_days=200,
        realized_annualized=8.0,
        expected_annualized=12.0,
        realized_max_dd=-25.0,  # worse than backtest's -16
        backtest_max_dd=-16.0,
    )
    assert read == "DRAWDOWN_BREACH"


def test_read_too_early_below_threshold() -> None:
    read, _ = pt._read(
        forward_trading_days=20,
        realized_annualized=50.0,
        expected_annualized=12.0,
        realized_max_dd=-5.0,
        backtest_max_dd=-16.0,
    )
    assert read == "TOO_EARLY"


def test_read_diverging_when_negative_vs_positive_expectation() -> None:
    read, _ = pt._read(
        forward_trading_days=200,
        realized_annualized=-8.0,
        expected_annualized=12.0,
        realized_max_dd=-10.0,
        backtest_max_dd=-16.0,
    )
    assert read == "DIVERGING"


def test_read_tracking_when_consistent() -> None:
    read, _ = pt._read(
        forward_trading_days=200,
        realized_annualized=11.0,
        expected_annualized=12.0,
        realized_max_dd=-10.0,
        backtest_max_dd=-16.0,
    )
    assert read == "TRACKING"


# ---- evaluate (mocked data + equity) ----


def _patch_eval(monkeypatch, equity: pd.Series) -> None:
    monkeypatch.setattr(
        pt, "load_portfolio_panel", lambda assets, start, end, min_rows=300: pd.DataFrame(
            {a: range(len(equity)) for a in assets}, index=equity.index
        )
    )
    monkeypatch.setattr(pt, "_forward_equity", lambda spec, panel: equity)


def test_evaluate_reports_no_data_when_nothing_after_inception(monkeypatch) -> None:
    # Equity ends before inception -> no forward bars.
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    equity = pd.Series(range(400), index=idx, dtype="float64") + 10_000
    _patch_eval(monkeypatch, equity)
    ev = pt.evaluate_paper_position(make_position(inception_date="2025-06-01"), as_of="2025-07-01")
    assert ev["status"] == "no_data_yet"


def test_evaluate_computes_forward_return(monkeypatch) -> None:
    idx = pd.date_range("2024-06-01", periods=500, freq="D")
    # Equity grows 10% over the forward window after inception 2025-01-01.
    equity = pd.Series(10_000.0, index=idx)
    inception_mask = idx >= pd.Timestamp("2025-01-01")
    # Set equity at inception to 10000 and final to 11000.
    fwd_len = inception_mask.sum()
    equity.loc[inception_mask] = [10_000.0 + 1000.0 * i / (fwd_len - 1) for i in range(fwd_len)]
    _patch_eval(monkeypatch, equity)

    ev = pt.evaluate_paper_position(make_position(inception_date="2025-01-01"), as_of="2025-12-31")

    assert ev["status"] == "evaluated"
    assert ev["realized_return_pct"] == pytest.approx(10.0, abs=0.5)
    assert ev["forward_trading_days"] == fwd_len
