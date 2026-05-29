import pytest

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.workflows import campaign_research


def make_spec(asset: str = "SPY", name: str = "S") -> StrategySpec:
    return StrategySpec(
        name=name,
        asset=asset,
        strategy_family=StrategyFamily.SMA_CROSSOVER,
        start_date="2020-01-01",
        end_date="2024-01-01",
        fast_window=50,
        slow_window=200,
        hypothesis="h",
    )


def make_backtest_result(spec: StrategySpec, beats: bool = False) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="vectorbt",
        metrics=BacktestMetrics(
            total_return_pct=10.0,
            buy_and_hold_return_pct=8.0,
            sharpe_ratio=1.0,
            max_drawdown_pct=-15.0,
            num_trades=30,
            win_rate_pct=55.0,
            exposure_time_pct=60.0,
            final_equity=11000.0,
            beats_benchmark=beats,
        ),
    )


def make_explore_result(asset: str, *, verdict: str = "needs_more_testing", lockbox_verdict: str | None = None, n_candidates: int = 3) -> dict:
    candidates = []
    for i in range(n_candidates):
        spec = make_spec(asset=asset, name=f"{asset}-{i}")
        candidates.append(
            {
                "strategy_spec": spec,
                "backtest_result": make_backtest_result(spec, beats=(verdict == "worth_paper_trading")),
                "report": ResearchReport(markdown="x", verdict=verdict, reasons=[], next_tests=[]),
            }
        )

    result: dict = {
        "candidates": candidates,
        "winner_index": 0,
        "winner_reason": "test",
        "failure_summary": {
            "verdict_counts": {verdict: n_candidates},
            "failed_check_counts": {"Benchmark comparison": n_candidates if verdict != "worth_paper_trading" else 0},
            "candidates_with_backtest": n_candidates,
            "candidates_without_backtest": 0,
        },
    }
    if lockbox_verdict is not None:
        lockbox_spec = make_spec(asset=asset, name=f"{asset}-lockbox")
        result["lockbox"] = {
            "strategy_spec": lockbox_spec,
            "backtest_result": make_backtest_result(lockbox_spec, beats=(lockbox_verdict == "worth_paper_trading")),
            "report": ResearchReport(markdown="x", verdict=lockbox_verdict, reasons=[], next_tests=[]),
        }
    return result


def test_campaign_runs_explore_per_asset(monkeypatch) -> None:
    invocations: list[tuple[str, int, float]] = []

    def fake_explore(user_request: str, slate_size: int, lockbox_pct: float = 0.0):
        # Extract the asset from the request (everything after the last "on ")
        asset = user_request.rsplit(" on ", 1)[-1]
        invocations.append((asset, slate_size, lockbox_pct))
        return make_explore_result(asset)

    monkeypatch.setattr(campaign_research, "run_exploration", fake_explore)

    result = campaign_research.run_campaign(
        idea="trend following",
        universe=["SPY", "QQQ", "BTC-USD"],
        slate_size=3,
        lockbox_pct=0.2,
    )

    assert [inv[0] for inv in invocations] == ["SPY", "QQQ", "BTC-USD"]
    assert all(inv[1] == 3 for inv in invocations)
    assert all(inv[2] == 0.2 for inv in invocations)
    assert len(result["slots"]) == 3
    assert result["summary"]["total_trials"] == 9


def test_campaign_dedupes_universe_case_insensitively(monkeypatch) -> None:
    invocations: list[str] = []

    def fake_explore(user_request: str, slate_size: int, lockbox_pct: float = 0.0):
        asset = user_request.rsplit(" on ", 1)[-1]
        invocations.append(asset)
        return make_explore_result(asset)

    monkeypatch.setattr(campaign_research, "run_exploration", fake_explore)

    result = campaign_research.run_campaign(
        idea="x",
        universe=["SPY", "spy", "  QQQ ", "QQQ", "BTC-USD"],
        slate_size=1,
    )

    assert invocations == ["SPY", "QQQ", "BTC-USD"]
    assert result["universe"] == ["SPY", "QQQ", "BTC-USD"]


def test_campaign_summarizes_per_asset_passes(monkeypatch) -> None:
    def fake_explore(user_request: str, slate_size: int, lockbox_pct: float = 0.0):
        asset = user_request.rsplit(" on ", 1)[-1]
        if asset == "SPY":
            return make_explore_result(asset, verdict="needs_more_testing", n_candidates=slate_size)
        if asset == "QQQ":
            return make_explore_result(
                asset,
                verdict="worth_paper_trading",
                lockbox_verdict="worth_paper_trading",
                n_candidates=slate_size,
            )
        if asset == "GLD":
            return make_explore_result(
                asset,
                verdict="worth_paper_trading",
                lockbox_verdict="needs_more_testing",
                n_candidates=slate_size,
            )
        return make_explore_result(asset, n_candidates=slate_size)

    monkeypatch.setattr(campaign_research, "run_exploration", fake_explore)

    result = campaign_research.run_campaign(
        idea="x", universe=["SPY", "QQQ", "GLD"], slate_size=2, lockbox_pct=0.2
    )

    summary = result["summary"]
    assert summary["assets_with_any_pass"] == ["QQQ", "GLD"]
    assert summary["assets_with_lockbox_pass"] == ["QQQ"]
    assert summary["total_trials"] == 6
    # SPY's stub has no lockbox; QQQ and GLD do.
    assert summary["total_lockbox_runs"] == 2


def test_campaign_captures_exploration_errors_per_asset(monkeypatch) -> None:
    def fake_explore(user_request: str, slate_size: int, lockbox_pct: float = 0.0):
        asset = user_request.rsplit(" on ", 1)[-1]
        if asset == "BAD":
            raise RuntimeError("network down")
        return make_explore_result(asset)

    monkeypatch.setattr(campaign_research, "run_exploration", fake_explore)

    result = campaign_research.run_campaign(
        idea="x", universe=["SPY", "BAD", "QQQ"], slate_size=1
    )

    slot_assets = [slot["asset"] for slot in result["slots"]]
    assert slot_assets == ["SPY", "BAD", "QQQ"]
    bad_slot = next(s for s in result["slots"] if s["asset"] == "BAD")
    assert "network down" in bad_slot["error"]
    # SPY and QQQ still ran
    assert result["summary"]["total_trials"] == 6  # 3 candidates each from SPY and QQQ


def test_campaign_rejects_empty_universe() -> None:
    with pytest.raises(ValueError):
        campaign_research.run_campaign(idea="x", universe=[], slate_size=3)


def test_campaign_rejects_zero_slate_size() -> None:
    with pytest.raises(ValueError):
        campaign_research.run_campaign(idea="x", universe=["SPY"], slate_size=0)


def test_campaign_aggregates_failed_checks(monkeypatch) -> None:
    def fake_explore(user_request: str, slate_size: int, lockbox_pct: float = 0.0):
        asset = user_request.rsplit(" on ", 1)[-1]
        return make_explore_result(asset, verdict="reject")  # all fail Benchmark

    monkeypatch.setattr(campaign_research, "run_exploration", fake_explore)

    result = campaign_research.run_campaign(
        idea="x", universe=["SPY", "QQQ"], slate_size=3
    )

    # 3 candidates per asset × 2 assets = 6 Benchmark failures aggregated
    assert result["summary"]["failed_checks"]["Benchmark comparison"] == 6
