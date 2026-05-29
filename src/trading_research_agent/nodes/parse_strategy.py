from langchain_core.messages import HumanMessage, SystemMessage

from trading_research_agent.config import load_settings
from trading_research_agent.schemas.strategy import StrategySpec


PARSER_PROMPT = """You are converting a natural-language trading research idea into a strict StrategySpec.

You must choose exactly one supported strategy family:
- sma_crossover
- donchian_breakout
- rsi_mean_reversion

Rules:
- Use daily data only.
- Use no leverage.
- Use long-only strategies.
- Use vectorbt as the backtest engine.
- Use coinmetrics as the data_source for BTC, BTC-USD, XBT, or bitcoin requests.
- Use fred as the data_source for Nasdaq, Nasdaq-100, NDX, or QQQ requests.
- Use yfinance as the data_source for other US equities and ETFs.
- Include realistic commission and slippage.
- Do not invent complex strategies.
- Do not claim profitability.
- Create a testable hypothesis.
- Prefer simple parameter values.

Default parameters:
- SMA crossover: fast_window=50, slow_window=200
- Donchian breakout: entry_window=55, exit_window=20
- RSI mean reversion: rsi_window=14, oversold_threshold=30, exit_threshold=50
"""


def parse_strategy_node(state: dict) -> dict:
    if state.get("strategy_spec") is not None:
        return {}

    user_request = state.get("user_request")
    if not user_request:
        return {"errors": _append_error(state, "Missing user_request")}

    try:
        settings = load_settings()
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.model,
            temperature=0,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        structured_model = model.with_structured_output(StrategySpec)
        parsed = structured_model.invoke(
            [
                SystemMessage(content=PARSER_PROMPT),
                HumanMessage(content=format_strategy_idea_prompt(user_request)),
            ]
        )
        if isinstance(parsed, StrategySpec):
            return {"strategy_spec": parsed}
        return {"strategy_spec": StrategySpec.model_validate(parsed)}
    except Exception as exc:
        return {"errors": _append_error(state, str(exc))}


def _append_error(state: dict, message: str) -> list[str]:
    return [*state.get("errors", []), message]


def format_strategy_idea_prompt(strategy: str) -> str:
    return f'Try this strategy: "{strategy}"'
