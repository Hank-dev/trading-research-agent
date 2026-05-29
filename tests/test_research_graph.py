import pytest

pytest.importorskip("langgraph")

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.workflows import research_graph


def make_spec(**overrides) -> StrategySpec:
    data = {
        "name": "Graph SMA",
        "asset": "SPY",
        "strategy_family": StrategyFamily.SMA_CROSSOVER,
        "start_date": "2018-01-01",
        "end_date": "2024-01-01",
        "fast_window": 50,
        "slow_window": 200,
        "hypothesis": "A simple trend-following rule may capture persistent trends.",
    }
    data.update(overrides)
    return StrategySpec(**data)


def fake_backtest_result(spec: StrategySpec) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="backtesting_py",
        metrics=BacktestMetrics(
            total_return_pct=12.0,
            buy_and_hold_return_pct=8.0,
            sharpe_ratio=1.1,
            max_drawdown_pct=-10.0,
            num_trades=25,
            win_rate_pct=52.0,
            exposure_time_pct=60.0,
            final_equity=11200.0,
            beats_benchmark=True,
        ),
    )


def test_valid_request_reaches_report_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest_node(state):
        return {"backtest_result": fake_backtest_result(state["strategy_spec"])}

    monkeypatch.setattr(research_graph, "run_backtest_node", fake_run_backtest_node)
    graph = research_graph.build_research_graph()

    state = graph.invoke(
        {
            "user_request": "Create a SPY SMA strategy.",
            "strategy_spec": make_spec(),
        }
    )

    assert state["report"].verdict == "worth_paper_trading"
    assert "backtest_result" in state


def test_invalid_strategy_does_not_run_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_run_backtest_node(state):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(research_graph, "run_backtest_node", fake_run_backtest_node)
    graph = research_graph.build_research_graph()

    state = graph.invoke(
        {
            "user_request": "Create a SPY SMA strategy.",
            "strategy_spec": make_spec(start_date="2024-01-01", end_date="2018-01-01"),
        }
    )

    assert called is False
    assert "backtest_result" not in state
    assert state["report"].verdict == "reject"


def test_rejected_strategy_produces_report_explaining_rejection() -> None:
    graph = research_graph.build_research_graph()

    state = graph.invoke(
        {
            "user_request": "Create a SPY SMA strategy.",
            "strategy_spec": make_spec(start_date="2024-01-01", end_date="2018-01-01"),
        }
    )

    assert "End date must be after start date" in state["report"].markdown


def test_graph_state_contains_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_backtest_node(state):
        return {"backtest_result": fake_backtest_result(state["strategy_spec"])}

    monkeypatch.setattr(research_graph, "run_backtest_node", fake_run_backtest_node)
    state = research_graph.build_research_graph().invoke(
        {
            "user_request": "Create a SPY SMA strategy.",
            "strategy_spec": make_spec(),
        }
    )

    assert {"strategy_spec", "critique", "backtest_result", "report"}.issubset(state)


def test_report_removes_banned_phrase_from_user_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_backtest_node(state):
        return {"backtest_result": fake_backtest_result(state["strategy_spec"])}

    monkeypatch.setattr(research_graph, "run_backtest_node", fake_run_backtest_node)
    state = research_graph.build_research_graph().invoke(
        {
            "user_request": "Find a profitable strategy for SPY.",
            "strategy_spec": make_spec(),
        }
    )

    assert "profitable strategy" not in state["report"].markdown.lower()
