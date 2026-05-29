import pytest

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.workflows import explore_research


def make_spec(name: str = "SMA", end_date: str = "2024-01-01") -> StrategySpec:
    return StrategySpec(
        name=name,
        asset="SPY",
        strategy_family=StrategyFamily.SMA_CROSSOVER,
        start_date="2020-01-01",
        end_date=end_date,
        fast_window=50,
        slow_window=200,
        hypothesis="A trend-following rule may capture persistent trends.",
    )


def make_backtest_result(
    spec: StrategySpec,
    sharpe: float | None = 1.0,
    total_return: float = 10.0,
    beats: bool = True,
) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="vectorbt",
        metrics=BacktestMetrics(
            total_return_pct=total_return,
            buy_and_hold_return_pct=5.0,
            sharpe_ratio=sharpe,
            max_drawdown_pct=-12.0,
            num_trades=25,
            win_rate_pct=55.0,
            exposure_time_pct=60.0,
            final_equity=11000.0,
            beats_benchmark=beats,
        ),
    )


def make_state(
    spec: StrategySpec,
    sharpe: float | None = 1.0,
    total_return: float = 10.0,
    beats: bool = True,
    verdict: str = "needs_more_testing",
) -> dict:
    return {
        "user_request": "Try SPY",
        "strategy_spec": spec,
        "backtest_result": make_backtest_result(
            spec, sharpe=sharpe, total_return=total_return, beats=beats
        ),
        "report": ResearchReport(
            markdown="# Report",
            verdict=verdict,
            reasons=[],
            next_tests=[],
        ),
    }


def test_split_date_range_basic() -> None:
    train_end, lockbox_start = split_date_range("2020-01-01", "2024-01-01", 0.2)
    assert train_end < lockbox_start
    assert lockbox_start <= "2024-01-01"
    assert train_end >= "2020-01-01"


def test_split_date_range_rejects_invalid_pct() -> None:
    with pytest.raises(ValueError):
        split_date_range("2020-01-01", "2024-01-01", 0.0)
    with pytest.raises(ValueError):
        split_date_range("2020-01-01", "2024-01-01", 1.0)


def test_split_date_range_rejects_short_range() -> None:
    with pytest.raises(ValueError):
        split_date_range("2024-01-01", "2024-01-01", 0.2)


def test_run_exploration_runs_every_candidate(monkeypatch) -> None:
    specs = [make_spec("Cand 1"), make_spec("Cand 2"), make_spec("Cand 3")]
    invoked_specs: list[StrategySpec] = []

    class FakeGraph:
        def invoke(self, state):
            invoked_specs.append(state["strategy_spec"])
            return make_state(state["strategy_spec"], sharpe=0.5)

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=3)

    assert [s.name for s in invoked_specs] == ["Cand 1", "Cand 2", "Cand 3"]
    assert len(result["candidates"]) == 3
    assert result["winner_index"] is not None
    assert "lockbox" not in result


def test_run_exploration_selects_highest_sharpe_among_beaters(monkeypatch) -> None:
    specs = [make_spec("Low"), make_spec("High"), make_spec("Mid")]
    sharpes = {"Low": 0.4, "High": 1.5, "Mid": 0.9}

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            return make_state(spec, sharpe=sharpes[spec.name], beats=True)

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=3)

    assert result["winner_index"] == 1
    assert result["winner_reason"] == "highest_sharpe_among_benchmark_beaters"


def test_run_exploration_falls_back_when_no_one_beats_benchmark(monkeypatch) -> None:
    specs = [make_spec("A"), make_spec("B")]
    sharpes = {"A": -0.5, "B": 0.2}

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            return make_state(spec, sharpe=sharpes[spec.name], beats=False)

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=2)

    assert result["winner_index"] == 1
    assert result["winner_reason"] == "highest_sharpe_overall_no_benchmark_beaters"


def test_run_exploration_with_lockbox_truncates_dates_and_reruns_winner(
    monkeypatch,
) -> None:
    specs = [make_spec("A"), make_spec("B")]
    sharpes = {"A": 1.2, "B": 0.4}
    seen_end_dates: list[str] = []
    seen_start_dates: list[str] = []

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            seen_end_dates.append(spec.end_date)
            seen_start_dates.append(spec.start_date)
            sharpe = sharpes.get(spec.name.replace(" (lockbox)", ""), 0.0)
            return make_state(spec, sharpe=sharpe)

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration(
        "Try SPY", slate_size=2, lockbox_pct=0.2
    )

    # First two invocations are the slate, on a truncated end date.
    assert seen_end_dates[0] == seen_end_dates[1]
    assert seen_end_dates[0] < "2024-01-01"
    assert seen_start_dates[0] == "2020-01-01"

    # Third invocation is the lockbox re-test of the winner ("A").
    assert "lockbox" in result
    assert result["lockbox"]["strategy_spec"].name == "A (lockbox)"
    assert seen_start_dates[2] > seen_end_dates[0]
    assert seen_end_dates[2] == "2024-01-01"

    assert result["lockbox_split"]["original_end"] == "2024-01-01"
    assert result["lockbox_split"]["train_end"] < result["lockbox_split"]["lockbox_start"]


def test_run_exploration_skips_lockbox_if_no_winner(monkeypatch) -> None:
    specs = [make_spec("A")]

    class FakeGraph:
        def invoke(self, state):
            return {"user_request": "x", "strategy_spec": state["strategy_spec"]}

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration(
        "Try SPY", slate_size=1, lockbox_pct=0.2
    )

    assert result["winner_index"] is None
    assert "lockbox" not in result


def test_run_exploration_handles_slate_generation_failure(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(explore_research, "generate_slate", boom)

    result = explore_research.run_exploration("Try SPY", slate_size=3)

    assert result["candidates"] == []
    assert result["winner_index"] is None
    assert any("Slate generation failed" in e for e in result.get("errors", []))


def test_run_exploration_rejects_zero_slate_size() -> None:
    with pytest.raises(ValueError):
        explore_research.run_exploration("Try SPY", slate_size=0)


def test_run_exploration_appends_dsr_check_to_every_candidate(monkeypatch) -> None:
    specs = [make_spec("A"), make_spec("B"), make_spec("C")]
    sharpes = {"A": 1.5, "B": 0.4, "C": -0.2}

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            return make_state(spec, sharpe=sharpes[spec.name])

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=3)

    for candidate in result["candidates"]:
        names = [c.test_name for c in candidate["backtest_result"].robustness_results]
        assert "Deflated Sharpe ratio (DSR)" in names


def test_dsr_check_says_insufficient_trials_when_only_one(monkeypatch) -> None:
    specs = [make_spec("Only")]

    class FakeGraph:
        def invoke(self, state):
            return make_state(state["strategy_spec"], sharpe=1.0)

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=1)

    dsr = next(
        check
        for check in result["candidates"][0]["backtest_result"].robustness_results
        if check.test_name == "Deflated Sharpe ratio (DSR)"
    )
    assert not dsr.passed
    assert "requires >= 2 trials" in dsr.details


def test_failure_summary_tallies_verdicts_and_failed_checks(monkeypatch) -> None:
    from trading_research_agent.schemas.backtest import RobustnessResult

    specs = [make_spec("A"), make_spec("B"), make_spec("C")]
    verdicts = {"A": "worth_paper_trading", "B": "needs_more_testing", "C": "reject"}
    check_pass = {
        "A": {"Benchmark comparison": True, "Positive return": True},
        "B": {"Benchmark comparison": False, "Positive return": True},
        "C": {"Benchmark comparison": False, "Positive return": False},
    }

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            base = make_state(spec, sharpe=0.5, verdict=verdicts[spec.name])
            base["backtest_result"].robustness_results = [
                RobustnessResult(test_name=name, passed=passed, details="x")
                for name, passed in check_pass[spec.name].items()
            ]
            return base

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=3)

    summary = result["failure_summary"]
    assert summary["candidates_with_backtest"] == 3
    assert summary["candidates_without_backtest"] == 0
    assert summary["verdict_counts"] == {
        "worth_paper_trading": 1,
        "needs_more_testing": 1,
        "reject": 1,
    }
    # Benchmark comparison failed twice (B, C); Positive return failed once (C).
    # DSR also failed for everyone because we mocked the graph (no robustness_checks_node ran)
    # — but DSR IS added by explore workflow, and Sharpe=0.5 with high variance fails the 0.95 gate.
    assert summary["failed_check_counts"]["Benchmark comparison"] == 2
    assert summary["failed_check_counts"]["Positive return"] == 1
    # Sorted by count descending — Benchmark should be first
    first_key = next(iter(summary["failed_check_counts"]))
    assert summary["failed_check_counts"][first_key] >= summary["failed_check_counts"]["Positive return"]


def test_failure_summary_counts_candidates_without_backtest(monkeypatch) -> None:
    specs = [make_spec("A"), make_spec("B")]

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            if spec.name == "A":
                return make_state(spec, sharpe=1.0, verdict="reject")
            return {"user_request": "x", "strategy_spec": spec}

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=2)

    summary = result["failure_summary"]
    assert summary["candidates_with_backtest"] == 1
    assert summary["candidates_without_backtest"] == 1


def test_dsr_is_lower_than_psr_when_slate_has_high_variance(monkeypatch) -> None:
    """DSR for the winner should be strictly lower than its per-trial PSR when
    other trials' Sharpes vary, i.e. multiple-testing actually penalizes."""
    from trading_research_agent.tools.stats import (
        estimate_trading_days,
        probabilistic_sharpe_ratio,
    )

    specs = [make_spec(name) for name in ["A", "B", "C", "D", "E"]]
    sharpes = {"A": 1.5, "B": 0.9, "C": 0.1, "D": -0.4, "E": 1.1}

    class FakeGraph:
        def invoke(self, state):
            spec = state["strategy_spec"]
            return make_state(spec, sharpe=sharpes[spec.name])

    monkeypatch.setattr(explore_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(explore_research, "generate_slate", lambda r, n: specs)

    result = explore_research.run_exploration("Try SPY", slate_size=5)
    winner = result["candidates"][result["winner_index"]]
    winner_result = winner["backtest_result"]
    dsr_check = next(
        c
        for c in winner_result.robustness_results
        if c.test_name == "Deflated Sharpe ratio (DSR)"
    )

    n_obs = estimate_trading_days(winner_result.start_date, winner_result.end_date)
    psr_value = probabilistic_sharpe_ratio(winner_result.metrics.sharpe_ratio, n_obs)

    dsr_value = float(dsr_check.details.split("DSR=")[1].split(" ")[0])
    assert dsr_value < psr_value
