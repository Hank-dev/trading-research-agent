import pandas as pd

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.workflows import robustness_stress as rs


def make_spec(**overrides) -> PortfolioSpec:
    data = {
        "name": "Winner",
        "assets": ["SPY", "TLT", "DBC", "GLD"],
        "portfolio_family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        "start_date": "2010-01-01",
        "end_date": "2024-01-01",
        "lookback_days": 126,
        "top_k": 2,
        "rebalance_days": 21,
        "hypothesis": "Cross-asset momentum.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


def _state(verdict: str) -> dict:
    spec = make_spec()
    return {
        "strategy_spec": spec,
        "backtest_result": BacktestResult(
            strategy_name="x",
            asset="PORTFOLIO[...]",
            start_date="2010-01-01",
            end_date="2024-01-01",
            engine="vectorbt_portfolio",
            metrics=BacktestMetrics(
                total_return_pct=30.0,
                buy_and_hold_return_pct=15.0,
                sharpe_ratio=0.9,
                max_drawdown_pct=-12.0,
                num_trades=40,
                win_rate_pct=55.0,
                exposure_time_pct=80.0,
                final_equity=13000.0,
                beats_benchmark=True,
            ),
        ),
        "report": ResearchReport(markdown="x", verdict=verdict, reasons=[], next_tests=[]),
    }


def _dummy_panel(spec: PortfolioSpec) -> pd.DataFrame:
    idx = pd.date_range("2010-01-01", periods=10, freq="D")
    return pd.DataFrame({a: range(10) for a in spec.assets}, index=idx)


def test_build_perturbations_covers_categories() -> None:
    perts = rs._build_perturbations(make_spec())
    categories = {cat for cat, _, _ in perts}
    assert "parameter" in categories
    assert "universe" in categories
    # Leave-one-out: one variant per asset (4 assets -> 4 drops).
    drops = [label for cat, label, _ in perts if cat == "universe"]
    assert len(drops) == 4
    # Parameter neighbors vary lookback, rebalance, and top_k.
    param_labels = " ".join(label for cat, label, _ in perts if cat == "parameter")
    assert "lookback=" in param_labels
    assert "rebalance=" in param_labels
    assert "top_k=" in param_labels


def test_leave_one_out_clamps_top_k_and_stays_valid() -> None:
    # 3 assets, top_k=3 -> dropping one leaves 2 assets; top_k must clamp to 2.
    spec = make_spec(assets=["SPY", "TLT", "GLD"], top_k=3)
    perts = rs._build_perturbations(spec)
    universe_variants = [v for cat, _, v in perts if cat == "universe"]
    assert universe_variants
    for v in universe_variants:
        assert len(v.assets) == 2
        assert v.top_k <= len(v.assets)


def test_run_stress_test_robust_when_everything_confirms(monkeypatch) -> None:
    spec = make_spec()
    monkeypatch.setattr(rs, "run_portfolio_backtest", lambda s, r, panel=None: _state("worth_paper_trading"))

    stress = rs.run_stress_test(spec, "2010-01-01", "2024-01-01", panel=_dummy_panel(spec))

    assert stress["summary"]["verdict"] == "ROBUST"
    assert stress["summary"]["overall_rate"] == 1.0
    assert stress["summary"]["base_confirms"] is True


def test_run_stress_test_broken_when_baseline_fails(monkeypatch) -> None:
    spec = make_spec()
    monkeypatch.setattr(rs, "run_portfolio_backtest", lambda s, r, panel=None: _state("needs_more_testing"))

    stress = rs.run_stress_test(spec, "2010-01-01", "2024-01-01", panel=_dummy_panel(spec))

    assert stress["summary"]["verdict"] == "BROKEN"
    assert stress["summary"]["base_confirms"] is False


def test_run_stress_test_fragile_when_universe_collapses(monkeypatch) -> None:
    spec = make_spec()

    def fake_backtest(s, r, panel=None):
        # Confirms for the full 4-asset universe; fails whenever an asset is dropped.
        if len(s.assets) < 4:
            return _state("needs_more_testing")
        return _state("worth_paper_trading")

    monkeypatch.setattr(rs, "run_portfolio_backtest", fake_backtest)

    stress = rs.run_stress_test(spec, "2010-01-01", "2024-01-01", panel=_dummy_panel(spec))

    summary = stress["summary"]
    assert summary["base_confirms"] is True
    # Universe category should have 0% survival -> overall verdict FRAGILE.
    assert summary["category_rates"]["universe"]["rate"] == 0.0
    assert summary["verdict"] == "FRAGILE"


def test_unrunnable_perturbations_are_marked(monkeypatch) -> None:
    spec = make_spec()

    def fake_backtest(s, r, panel=None):
        # Lockbox specs (held-out) come back with no backtest -> unrunnable.
        if "(lockbox)" in s.name:
            return {"strategy_spec": s, "report": None, "errors": ["panel too short"]}
        return _state("worth_paper_trading")

    monkeypatch.setattr(rs, "run_portfolio_backtest", fake_backtest)

    stress = rs.run_stress_test(spec, "2010-01-01", "2024-01-01", panel=_dummy_panel(spec))

    assert stress["summary"]["unrunnable"] > 0
    assert all(not r["confirms"] for r in stress["results"] if r["status"] != "ok")


# ---- history reconstruction ----


def test_latest_confirmed_portfolio_winner_joins_by_slate_id() -> None:
    records = [
        {
            "mode": "portfolio",
            "slate_id": "aaa",
            "is_lockbox": False,
            "timestamp": "2026-05-29T10:00:00Z",
            "strategy_family": "cross_sectional_momentum",
            "start_date": "2010-01-01",
            "end_date": "2021-04-01",
            "verdict": "worth_paper_trading",
            "params": {"assets": ["SPY", "TLT", "DBC", "GLD"], "lookback_days": 126, "top_k": 2, "rebalance_days": 21},
        },
        {
            "mode": "portfolio",
            "slate_id": "aaa",
            "is_lockbox": True,
            "timestamp": "2026-05-29T10:00:05Z",
            "strategy_family": "cross_sectional_momentum",
            "start_date": "2021-04-02",
            "end_date": "2024-01-01",
            "verdict": "worth_paper_trading",
            "params": {"assets": ["SPY", "TLT", "DBC", "GLD"], "lookback_days": 126, "top_k": 2, "rebalance_days": 21},
        },
    ]
    winner = rs.latest_confirmed_portfolio_winner(records)
    assert winner is not None
    assert winner["full_start"] == "2010-01-01"
    assert winner["full_end"] == "2024-01-01"
    assert winner["strategy_family"] == "cross_sectional_momentum"

    spec = rs.spec_from_winner(winner)
    assert spec.assets == ["SPY", "TLT", "DBC", "GLD"]
    assert spec.top_k == 2
    assert spec.start_date == "2010-01-01"
    assert spec.end_date == "2024-01-01"


def test_latest_confirmed_winner_none_when_no_lockbox_pass() -> None:
    records = [
        {
            "mode": "portfolio",
            "slate_id": "bbb",
            "is_lockbox": True,
            "timestamp": "2026-05-29T10:00:05Z",
            "strategy_family": "dual_momentum",
            "start_date": "2021-01-01",
            "end_date": "2024-01-01",
            "verdict": "needs_more_testing",
            "params": {"assets": ["SPY", "TLT"], "lookback_days": 126, "top_k": 1, "rebalance_days": 21},
        }
    ]
    assert rs.latest_confirmed_portfolio_winner(records) is None


def test_latest_confirmed_winner_picks_most_recent() -> None:
    def group(slate_id, ts, verdict):
        return [
            {
                "mode": "portfolio", "slate_id": slate_id, "is_lockbox": False,
                "timestamp": ts, "strategy_family": "cross_sectional_momentum",
                "start_date": "2010-01-01", "end_date": "2021-01-01", "verdict": "worth_paper_trading",
                "params": {"assets": ["SPY", "TLT", "GLD"], "lookback_days": 100, "top_k": 1, "rebalance_days": 21},
            },
            {
                "mode": "portfolio", "slate_id": slate_id, "is_lockbox": True,
                "timestamp": ts, "strategy_family": "cross_sectional_momentum",
                "start_date": "2021-01-02", "end_date": "2024-01-01", "verdict": verdict,
                "params": {"assets": ["SPY", "TLT", "GLD"], "lookback_days": 100, "top_k": 1, "rebalance_days": 21},
            },
        ]

    records = group("old", "2026-05-01T00:00:00Z", "worth_paper_trading") + group(
        "new", "2026-05-28T00:00:00Z", "worth_paper_trading"
    )
    winner = rs.latest_confirmed_portfolio_winner(records)
    assert winner["timestamp"] == "2026-05-28T00:00:00Z"
