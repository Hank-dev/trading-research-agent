from typing import TypedDict

from trading_research_agent.nodes.critique_strategy import critique_strategy_node
from trading_research_agent.nodes.generate_report import generate_report_node
from trading_research_agent.nodes.parse_strategy import parse_strategy_node
from trading_research_agent.nodes.robustness_checks import robustness_checks_node
from trading_research_agent.nodes.run_backtest import run_backtest_node
from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategySpec


class ResearchState(TypedDict, total=False):
    user_request: str
    strategy_spec: StrategySpec
    critique: StrategyCritique
    backtest_result: BacktestResult
    report: ResearchReport
    errors: list[str]


def build_research_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ResearchState)
    graph.add_node("parse_strategy", parse_strategy_node)
    graph.add_node("critique_strategy", critique_strategy_node)
    graph.add_node("run_backtest", run_backtest_node)
    graph.add_node("robustness_checks", robustness_checks_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("parse_strategy")
    graph.add_edge("parse_strategy", "critique_strategy")
    graph.add_conditional_edges(
        "critique_strategy",
        _route_after_critique,
        {
            "run_backtest": "run_backtest",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("run_backtest", "robustness_checks")
    graph.add_edge("robustness_checks", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()


def _route_after_critique(state: ResearchState) -> str:
    critique = state.get("critique")
    if critique is not None and critique.approved:
        return "run_backtest"
    return "generate_report"


def run_research(user_request: str) -> ResearchState:
    graph = build_research_graph()
    return graph.invoke({"user_request": user_request})
