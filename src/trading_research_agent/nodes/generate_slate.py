from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trading_research_agent.config import load_settings
from trading_research_agent.schemas.strategy import StrategySpec


SLATE_PROMPT = """You are pre-registering a slate of distinct trading research hypotheses BEFORE any backtest runs.

Rules:
- Produce exactly the requested number of StrategySpec entries.
- Each entry must be a different, defensible hypothesis. Prefer DIFFERENT strategy_family values over parameter tweaks of the same family.
- Cover at least two of: sma_crossover, donchian_breakout, rsi_mean_reversion when the slate has size >= 2.
- Use the same asset, data_source, timeframe, date range, initial cash, commission, slippage, and benchmark across every entry.
- Use daily data only, long-only, no leverage, vectorbt as the backtest engine.
- For BTC, BTC-USD, XBT, or bitcoin requests, use data_source=coinmetrics.
- For Nasdaq, Nasdaq-100, NDX, or QQQ requests, use data_source=fred.
- Use yfinance for other US equities and ETFs.
- Include realistic commission and slippage.
- Do not invent performance metrics or claim profitability.
- Each entry must have a distinct, plain-English hypothesis.

This is a pre-registered set: you cannot revise the slate after seeing results.
"""


class StrategySlate(BaseModel):
    """Wrapper schema so structured output can return a list of specs."""

    strategies: list[StrategySpec] = Field(min_length=1)


def generate_slate(user_request: str, slate_size: int) -> list[StrategySpec]:
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1")

    settings = load_settings()
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.model,
        temperature=0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    structured_model = model.with_structured_output(StrategySlate)
    parsed = structured_model.invoke(
        [
            SystemMessage(content=SLATE_PROMPT),
            HumanMessage(content=_format_slate_prompt(user_request, slate_size)),
        ]
    )
    slate = parsed if isinstance(parsed, StrategySlate) else StrategySlate.model_validate(parsed)
    if not slate.strategies:
        raise ValueError("Slate generation returned no strategies")
    return slate.strategies[:slate_size]


def _format_slate_prompt(user_request: str, slate_size: int) -> str:
    return (
        f'Original research idea: "{user_request}"\n\n'
        f"Produce a pre-registered slate of {slate_size} distinct StrategySpec entries."
    )
