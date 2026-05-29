from langchain_core.messages import HumanMessage, SystemMessage

from trading_research_agent.config import load_settings
from trading_research_agent.schemas.strategy import StrategySpec


REFINEMENT_PROMPT = """You are proposing exactly one follow-up StrategySpec after seeing a deterministic backtest result.

Rules:
- Create one new testable strategy specification.
- Keep the same asset, data_source, timeframe, backtest_engine, date range, initial cash, commission, slippage, and benchmark unless the previous spec used a wrong BTC data source.
- Prefer backtest_engine=vectorbt when proposing a new spec so walk-forward and Monte Carlo robustness checks are available.
- For BTC, BTC-USD, XBT, or bitcoin, use data_source=coinmetrics.
- Choose exactly one supported family: sma_crossover, donchian_breakout, or rsi_mean_reversion.
- Do not invent performance metrics.
- Do not claim profitability.
- Do not optimize until results look good.
- Avoid tiny parameter tweaks whose only purpose is improving the prior result.
- Prefer a simple, defensible alternative hypothesis based on the prior failure modes.
- The output must be a valid StrategySpec.
"""


def refine_strategy_node(state: dict) -> dict:
    if state.get("backtest_result") is None:
        return {"errors": _append_error(state, "Cannot refine without backtest_result")}

    try:
        return {"iteration_strategy_spec": propose_next_strategy(state)}
    except Exception as exc:
        return {"errors": _append_error(state, f"Strategy refinement failed: {exc}")}


def propose_next_strategy(state: dict) -> StrategySpec:
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
            SystemMessage(content=REFINEMENT_PROMPT),
            HumanMessage(content=format_refinement_context(state)),
        ]
    )
    if isinstance(parsed, StrategySpec):
        return parsed
    return StrategySpec.model_validate(parsed)


def format_refinement_context(state: dict) -> str:
    spec = state.get("strategy_spec")
    critique = state.get("critique")
    result = state.get("backtest_result")
    report = state.get("report")

    sections = [
        f"Original user request:\n{state.get('user_request', '')}",
        f"Previous StrategySpec:\n{_dump_model(spec)}",
        f"Previous critique:\n{_dump_model(critique)}",
        f"Previous BacktestResult:\n{_dump_model(result)}",
        f"Previous verdict:\n{getattr(report, 'verdict', 'unavailable')}",
        (
            "Task:\nPropose one revised StrategySpec and nothing else. "
            "The revised strategy will be fed back into the same critique and "
            "backtest pipeline for one additional iteration."
        ),
    ]
    return "\n\n".join(sections)


def _dump_model(value: object) -> str:
    if value is None:
        return "unavailable"
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(indent=2)
    return str(value)


def _append_error(state: dict, message: str) -> list[str]:
    return [*state.get("errors", []), message]
