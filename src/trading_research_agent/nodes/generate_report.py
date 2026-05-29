from trading_research_agent.reports.markdown_report import build_research_report


def generate_report_node(state: dict) -> dict:
    report = build_research_report(
        user_request=state.get("user_request"),
        strategy_spec=state.get("strategy_spec"),
        critique=state.get("critique"),
        backtest_result=state.get("backtest_result"),
        errors=state.get("errors", []),
    )
    return {"report": report}
