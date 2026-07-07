from pathlib import Path


from trading_research_agent.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    RobustnessResult,
)
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.tools import history


def make_spec(name: str = "X", family: StrategyFamily = StrategyFamily.SMA_CROSSOVER) -> StrategySpec:
    kwargs: dict = {
        "name": name,
        "asset": "BTC-USD",
        "strategy_family": family,
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "hypothesis": "h",
    }
    if family == StrategyFamily.SMA_CROSSOVER:
        kwargs.update(fast_window=50, slow_window=200)
    elif family == StrategyFamily.DONCHIAN_BREAKOUT:
        kwargs.update(entry_window=55, exit_window=20)
    elif family == StrategyFamily.RSI_MEAN_REVERSION:
        kwargs.update(rsi_window=14, oversold_threshold=30, exit_threshold=50)
    return StrategySpec(**kwargs)


def make_state(
    spec: StrategySpec,
    *,
    verdict: str = "needs_more_testing",
    sharpe: float | None = 0.9,
    beats: bool = False,
    fail_checks: list[str] | None = None,
) -> dict:
    fail_checks = fail_checks or []
    return {
        "strategy_spec": spec,
        "backtest_result": BacktestResult(
            strategy_name=spec.name,
            asset=spec.asset,
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine="vectorbt",
            metrics=BacktestMetrics(
                total_return_pct=5.0,
                buy_and_hold_return_pct=10.0,
                sharpe_ratio=sharpe,
                max_drawdown_pct=-15.0,
                num_trades=30,
                win_rate_pct=50.0,
                exposure_time_pct=60.0,
                final_equity=10500.0,
                beats_benchmark=beats,
            ),
            robustness_results=[
                RobustnessResult(test_name=name, passed=False, details="x")
                for name in fail_checks
            ],
        ),
        "report": ResearchReport(markdown="x", verdict=verdict, reasons=[], next_tests=[]),
    }


def test_record_from_state_extracts_expected_fields() -> None:
    spec = make_spec("SMA")
    state = make_state(spec, verdict="reject", fail_checks=["Benchmark comparison"])

    record = history.record_from_state(
        state, mode="explore", user_request="Try BTC", slate_id="abc123"
    )

    assert record is not None
    assert record["mode"] == "explore"
    assert record["user_request"] == "Try BTC"
    assert record["slate_id"] == "abc123"
    assert record["is_lockbox"] is False
    assert record["asset"] == "BTC-USD"
    assert record["strategy_family"] == "sma_crossover"
    assert record["params"] == {"fast_window": 50, "slow_window": 200}
    assert record["verdict"] == "reject"
    assert record["failed_checks"] == ["Benchmark comparison"]
    assert record["metrics"]["sharpe_ratio"] == 0.9
    assert "timestamp" in record


def test_record_from_state_omits_failed_checks_when_none() -> None:
    spec = make_spec("X")
    state = make_state(spec, fail_checks=[])
    record = history.record_from_state(state, mode="single", user_request="x")
    assert "failed_checks" not in record


def test_record_from_state_handles_donchian_params() -> None:
    spec = make_spec("Don", family=StrategyFamily.DONCHIAN_BREAKOUT)
    state = make_state(spec)
    record = history.record_from_state(state, mode="single", user_request="x")
    assert record["params"] == {"entry_window": 55, "exit_window": 20}


def test_record_from_state_returns_none_for_empty_state() -> None:
    assert history.record_from_state({}, mode="single", user_request="x") is None


def test_record_from_state_handles_portfolio_spec() -> None:
    from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec

    spec = PortfolioSpec(
        name="Rotation",
        assets=["SPY", "TLT", "GLD"],
        portfolio_family=PortfolioFamily.DUAL_MOMENTUM,
        start_date="2015-01-01",
        end_date="2023-01-01",
        lookback_days=126,
        top_k=1,
        rebalance_days=21,
        hypothesis="Cross-asset dual momentum.",
    )
    state = {
        "strategy_spec": spec,
        "report": ResearchReport(markdown="x", verdict="needs_more_testing", reasons=[], next_tests=[]),
    }
    record = history.record_from_state(state, mode="portfolio", user_request="rotate")

    assert record is not None
    assert record["asset"] == "PORTFOLIO[SPY,TLT,GLD]"
    assert record["strategy_family"] == "dual_momentum"
    assert record["data_source"] == "auto"
    assert record["params"]["assets"] == ["SPY", "TLT", "GLD"]
    assert record["params"]["lookback_days"] == 126
    assert record["params"]["top_k"] == 1


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    history.append_run_record({"asset": "BTC", "verdict": "reject"}, path=path)
    history.append_run_record({"asset": "QQQ", "verdict": "needs_more_testing"}, path=path)

    loaded = history.load_history(path=path)
    assert len(loaded) == 2
    assert loaded[0]["asset"] == "BTC"
    assert loaded[1]["asset"] == "QQQ"


def test_default_history_path_uses_configured_output_dir(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "vps-output"
    monkeypatch.setenv("TRADING_RESEARCH_OUTPUT_DIR", str(output_dir))

    history.append_run_record({"asset": "BTC", "verdict": "reject"})

    loaded = history.load_history()
    assert loaded == [{"asset": "BTC", "verdict": "reject"}]
    assert (output_dir / "history.jsonl").exists()


def test_load_history_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert history.load_history(path=tmp_path / "nope.jsonl") == []


def test_load_history_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "h.jsonl"
    path.write_text('{"asset": "BTC"}\nnot-json\n{"asset": "QQQ"}\n')
    loaded = history.load_history(path=path)
    assert [r["asset"] for r in loaded] == ["BTC", "QQQ"]


def test_summarize_history_tallies_correctly() -> None:
    records = [
        {
            "timestamp": "2026-05-01T00:00:00Z",
            "is_lockbox": False,
            "asset": "BTC",
            "strategy_family": "sma_crossover",
            "verdict": "reject",
            "failed_checks": ["Benchmark comparison", "Deflated Sharpe ratio (DSR)"],
        },
        {
            "timestamp": "2026-05-02T00:00:00Z",
            "is_lockbox": False,
            "asset": "BTC",
            "strategy_family": "donchian_breakout",
            "verdict": "needs_more_testing",
            "failed_checks": ["Deflated Sharpe ratio (DSR)"],
        },
        {
            "timestamp": "2026-05-03T00:00:00Z",
            "is_lockbox": False,
            "asset": "QQQ",
            "strategy_family": "sma_crossover",
            "verdict": "worth_paper_trading",
            "failed_checks": [],
        },
        {
            "timestamp": "2026-05-04T00:00:00Z",
            "is_lockbox": True,
            "asset": "QQQ",
            "strategy_family": "sma_crossover",
            "verdict": "needs_more_testing",
        },
        {
            "timestamp": "2026-05-05T00:00:00Z",
            "mode": "stress",
            "is_lockbox": False,
            "asset": "QQQ",
            "strategy_family": "sma_crossover",
            "verdict": "ROBUST",
        },
    ]
    summary = history.summarize_history(records)

    assert summary["total_trials"] == 3
    assert summary["lockbox_runs"] == 1
    assert summary["stress_runs"] == 1
    assert summary["robust_stress_runs"] == 1
    assert summary["by_asset"] == {"BTC": 2, "QQQ": 1}
    assert summary["by_family"] == {"sma_crossover": 2, "donchian_breakout": 1}
    assert summary["by_verdict"]["reject"] == 1
    assert summary["by_verdict"]["worth_paper_trading"] == 1
    assert summary["failed_checks"]["Deflated Sharpe ratio (DSR)"] == 2
    assert summary["failed_checks"]["Benchmark comparison"] == 1
    assert len(summary["passed_runs"]) == 1
    assert summary["passed_runs"][0]["asset"] == "QQQ"


def test_summarize_history_tracks_structured_learnings() -> None:
    records = [
        {
            "timestamp": "2026-05-06T00:00:00Z",
            "mode": "event_followthrough",
            "is_lockbox": False,
            "asset": "BTC-USD",
            "strategy_family": "event_followthrough",
            "verdict": "winner",
            "learning_status": "winner",
            "lesson": "UUP weakness -> BTC survived stress.",
        },
        {
            "timestamp": "2026-05-07T00:00:00Z",
            "mode": "event_followthrough",
            "is_lockbox": False,
            "asset": "BTC-USD",
            "strategy_family": "event_followthrough",
            "verdict": "lockbox_loser",
            "learning_status": "lockbox_loser",
            "lesson": "TLT strength -> BTC failed lockbox.",
        },
    ]

    summary = history.summarize_history(records)
    assert summary["by_learning_status"] == {"winner": 1, "lockbox_loser": 1}
    assert summary["learning_records"][0]["timestamp"] == "2026-05-07T00:00:00Z"
    text = history.format_summary(summary)
    assert "Structured learnings" in text
    assert "Recent lessons" in text
    assert "UUP weakness" in text


def test_summarize_history_handles_empty() -> None:
    summary = history.summarize_history([])
    assert summary["total_trials"] == 0
    assert summary["lockbox_runs"] == 0
    assert summary["by_asset"] == {}
    assert summary["passed_runs"] == []


def test_format_summary_with_no_history_explains_what_to_do() -> None:
    text = history.format_summary(history.summarize_history([]))
    assert "No history yet" in text


def test_format_summary_includes_passed_runs_section() -> None:
    records = [
        {
            "timestamp": "2026-05-03T00:00:00Z",
            "asset": "QQQ",
            "strategy_family": "sma_crossover",
            "verdict": "worth_paper_trading",
        }
    ]
    text = history.format_summary(history.summarize_history(records))
    assert "QQQ sma_crossover" in text
    assert "Cross-run multiple-testing note" in text
