from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.workflows import iterative_research


def make_spec(name: str = "SMA") -> StrategySpec:
    return StrategySpec(
        name=name,
        asset="SPY",
        strategy_family=StrategyFamily.SMA_CROSSOVER,
        start_date="2020-01-01",
        end_date="2024-01-01",
        fast_window=50,
        slow_window=200,
        hypothesis="A trend-following rule may capture persistent trends.",
    )


def make_backtest_result(spec: StrategySpec) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="backtesting_py",
        metrics=BacktestMetrics(
            total_return_pct=5.0,
            buy_and_hold_return_pct=3.0,
            sharpe_ratio=0.8,
            max_drawdown_pct=-12.0,
            num_trades=12,
            win_rate_pct=50.0,
            exposure_time_pct=60.0,
            final_equity=10500.0,
            beats_benchmark=True,
        ),
    )


def make_state(spec: StrategySpec, verdict: str = "needs_more_testing") -> dict:
    return {
        "user_request": "Try SPY SMA",
        "strategy_spec": spec,
        "backtest_result": make_backtest_result(spec),
        "report": ResearchReport(
            markdown="# Report",
            verdict=verdict,
            reasons=[],
            next_tests=[],
        ),
    }


def test_run_research_with_one_iteration_feeds_next_spec_back(monkeypatch) -> None:
    initial_spec = make_spec("Initial")
    next_spec = make_spec("Iteration")
    calls: list[dict] = []

    class FakeGraph:
        def invoke(self, state):
            calls.append(state)
            if "strategy_spec" in state:
                return make_state(state["strategy_spec"])
            return make_state(initial_spec)

    monkeypatch.setattr(iterative_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        iterative_research,
        "propose_next_strategy",
        lambda state: next_spec,
    )

    result = iterative_research.run_research_with_one_iteration("Try SPY SMA")

    assert result["initial"]["strategy_spec"].name == "Initial"
    assert result["iteration"]["strategy_spec"].name == "Iteration"
    assert result["iterations"][-1]["strategy_spec"].name == "Iteration"
    assert calls[1]["strategy_spec"] == next_spec


def test_run_research_with_one_iteration_skips_if_initial_backtest_missing(
    monkeypatch,
) -> None:
    class FakeGraph:
        def invoke(self, state):
            return {"user_request": state["user_request"]}

    monkeypatch.setattr(iterative_research, "build_research_graph", lambda: FakeGraph())

    result = iterative_research.run_research_with_one_iteration("Try SPY SMA")

    assert "iteration" not in result
    assert result["errors"] == [
        "Initial backtest did not run; skipping strategy iteration."
    ]


def test_run_research_until_pass_stops_when_iteration_passes(monkeypatch) -> None:
    initial_spec = make_spec("Initial")
    candidate_specs = [make_spec("Candidate 1"), make_spec("Candidate 2")]
    proposed = iter(candidate_specs)
    graph_calls: list[dict] = []

    class FakeGraph:
        def invoke(self, state):
            graph_calls.append(state)
            if "strategy_spec" not in state:
                return make_state(initial_spec, verdict="needs_more_testing")
            verdict = (
                "worth_paper_trading"
                if state["strategy_spec"].name == "Candidate 2"
                else "needs_more_testing"
            )
            return make_state(state["strategy_spec"], verdict=verdict)

    monkeypatch.setattr(iterative_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        iterative_research,
        "propose_next_strategy",
        lambda state: next(proposed),
    )

    result = iterative_research.run_research_until_pass(
        "Try SPY SMA", max_iterations=5
    )

    assert result["passed"] is True
    assert result["stop_reason"] == "passed_iteration_2"
    assert len(result["iterations"]) == 2
    assert len(graph_calls) == 3


def test_run_research_until_pass_stops_at_max_iterations(monkeypatch) -> None:
    initial_spec = make_spec("Initial")
    counter = {"value": 0}

    class FakeGraph:
        def invoke(self, state):
            if "strategy_spec" not in state:
                return make_state(initial_spec, verdict="needs_more_testing")
            return make_state(state["strategy_spec"], verdict="needs_more_testing")

    def fake_propose_next_strategy(state):
        counter["value"] += 1
        return make_spec(f"Candidate {counter['value']}")

    monkeypatch.setattr(iterative_research, "build_research_graph", lambda: FakeGraph())
    monkeypatch.setattr(
        iterative_research,
        "propose_next_strategy",
        fake_propose_next_strategy,
    )

    result = iterative_research.run_research_until_pass(
        "Try SPY SMA", max_iterations=3
    )

    assert result["passed"] is False
    assert result["stop_reason"] == "max_iterations_reached"
    assert len(result["iterations"]) == 3


def test_run_research_until_pass_does_not_iterate_if_initial_passes(monkeypatch) -> None:
    initial_spec = make_spec("Initial")

    class FakeGraph:
        def invoke(self, state):
            return make_state(initial_spec, verdict="worth_paper_trading")

    monkeypatch.setattr(iterative_research, "build_research_graph", lambda: FakeGraph())

    result = iterative_research.run_research_until_pass(
        "Try SPY SMA", max_iterations=5
    )

    assert result["passed"] is True
    assert result["stop_reason"] == "initial_passed"
    assert result["iterations"] == []
