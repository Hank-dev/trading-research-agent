from typing import TypedDict

from trading_research_agent.nodes.refine_strategy import propose_next_strategy
from trading_research_agent.workflows.research_graph import ResearchState, build_research_graph


class IterativeResearchResult(TypedDict, total=False):
    initial: ResearchState
    iteration: ResearchState
    iterations: list[ResearchState]
    passed: bool
    stop_reason: str
    errors: list[str]


def run_research_with_one_iteration(user_request: str) -> IterativeResearchResult:
    result = run_research_until_pass(user_request, max_iterations=1)
    if result.get("iterations"):
        result["iteration"] = result["iterations"][-1]
    return result


def run_research_until_pass(
    user_request: str,
    max_iterations: int = 5,
) -> IterativeResearchResult:
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")

    graph = build_research_graph()
    initial = graph.invoke({"user_request": user_request})
    if initial.get("backtest_result") is None:
        return {
            "initial": initial,
            "iterations": [],
            "passed": False,
            "stop_reason": "initial_backtest_missing",
            "errors": ["Initial backtest did not run; skipping strategy iteration."],
        }
    if _passed(initial):
        return {
            "initial": initial,
            "iterations": [],
            "passed": True,
            "stop_reason": "initial_passed",
        }

    iterations: list[ResearchState] = []
    current = initial
    for index in range(max_iterations):
        try:
            next_spec = propose_next_strategy(current)
        except Exception as exc:
            return {
                "initial": initial,
                "iterations": iterations,
                "passed": False,
                "stop_reason": "refinement_failed",
                "errors": [f"Strategy iteration failed: {exc}"],
            }

        iteration = graph.invoke(
            {
                "user_request": _iteration_request(user_request, index + 1),
                "strategy_spec": next_spec,
            }
        )
        iterations.append(iteration)
        if _passed(iteration):
            return {
                "initial": initial,
                "iterations": iterations,
                "passed": True,
                "stop_reason": f"passed_iteration_{index + 1}",
            }
        if iteration.get("backtest_result") is None:
            return {
                "initial": initial,
                "iterations": iterations,
                "passed": False,
                "stop_reason": f"iteration_{index + 1}_backtest_missing",
                "errors": [f"Iteration {index + 1} backtest did not run."],
            }
        current = iteration

    return {
        "initial": initial,
        "iterations": iterations,
        "passed": False,
        "stop_reason": "max_iterations_reached",
    }


def _passed(state: ResearchState) -> bool:
    report = state.get("report")
    return report is not None and report.verdict == "worth_paper_trading"


def _iteration_request(user_request: str, iteration_number: int) -> str:
    return (
        f"{user_request}\n\n"
        f"Iteration {iteration_number}: refinement generated after reviewing "
        "the previous deterministic backtest."
    )
