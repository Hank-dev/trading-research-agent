from trading_research_agent.nodes.parse_strategy import format_strategy_idea_prompt


def test_strategy_idea_prompt_wraps_user_strategy() -> None:
    prompt = format_strategy_idea_prompt("Buy BTC when the 50 SMA crosses above 200 SMA")

    assert prompt == 'Try this strategy: "Buy BTC when the 50 SMA crosses above 200 SMA"'
