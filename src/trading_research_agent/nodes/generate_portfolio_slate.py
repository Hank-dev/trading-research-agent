from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trading_research_agent.config import load_settings
from trading_research_agent.schemas.portfolio import PortfolioSpec


SLATE_PROMPT = """You are pre-registering a slate of distinct MULTI-ASSET PORTFOLIO research hypotheses BEFORE any backtest runs.

You must choose exactly one portfolio_family per entry:
- cross_sectional_momentum: hold the top_k assets by trailing return, rebalanced periodically.
- dual_momentum: cross-sectional momentum, but only hold assets with positive trailing return; otherwise that allocation sits in cash.
- equal_weight_trend: equal-weight each asset, but only hold it while above its own moving average.
- time_series_momentum: AQR-style; hold each asset's equal-weight slice only while its own trailing return is positive, else cash. Good for crisis-defensive baskets spanning equities, bonds, gold, commodities.
- crisis_hedge: EXACTLY 2 assets [core_risk_asset, volatility_hedge]. Hold the core while above its own SMA; on a break below, exit to cash and hold a capped `hedge_weight` slice of the volatility hedge. Requires hedge_weight in (0, 0.5]. Use ONLY for explicit crisis-hedge / tail-protection requests.

Crisis-hedge / volatility guidance:
- The volatility hedge should be a long-volatility ETF such as VIXY or UVXY. These ETFs bleed heavily in calm markets (negative roll yield), so crisis_hedge only holds them during risk-off regimes.
- Keep hedge_weight small (0.1-0.25). Volatility ETFs are not portfolio-sized positions.
- Do not use SVXY or other SHORT-volatility products (they can collapse, e.g. Feb 2018).
- A crisis hedge legitimately gives up return for drawdown protection; do not expect it to beat its core asset on raw return.

Rules:
- Produce exactly the requested number of PortfolioSpec entries.
- Each entry must be a STRUCTURALLY DIFFERENT hypothesis: vary the family, the universe, or the lookback/rebalance regime. Do NOT submit near-duplicates that differ only by a small parameter.
- Each universe must have at least 2 assets and they must share overlapping history.
- Use daily data only (timeframe=1d) and the vectorbt engine.
- ASSET AVAILABILITY (important): only use assets the data layer can load:
  - BTC-USD (crypto) loads from Coin Metrics.
  - QQQ / Nasdaq-100 loads from FRED.
  - Liquid US ETFs and US stocks (e.g. SPY, IWM, EFA, EEM, TLT, IEF, LQD, HYG, GLD, SLV, DBC, XLU, XLP, XLE, XLF, XLK) load from Tiingo.
  - Do NOT use spot commodities, VIX indices, or futures symbols; use the corresponding ETF instead (e.g. GLD for gold, DBC for broad commodities).
- Prefer universes that span different asset classes (equities, bonds, commodities) so rotation has something to rotate into.
- lookback_days between 20 and 504; rebalance_days between 1 and 252 (21 ~ monthly); top_k between 1 and the universe size.
- Do not claim profitability. Each entry must have a distinct, plain-English hypothesis.

This is a pre-registered set: you cannot revise the slate after seeing results.
"""


class PortfolioSlate(BaseModel):
    """Wrapper so structured output can return a list of portfolio specs."""

    portfolios: list[PortfolioSpec] = Field(min_length=1)


def generate_portfolio_slate(user_request: str, slate_size: int) -> list[PortfolioSpec]:
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
    structured_model = model.with_structured_output(PortfolioSlate)
    parsed = structured_model.invoke(
        [
            SystemMessage(content=SLATE_PROMPT),
            HumanMessage(content=_format_prompt(user_request, slate_size)),
        ]
    )
    slate = parsed if isinstance(parsed, PortfolioSlate) else PortfolioSlate.model_validate(parsed)
    if not slate.portfolios:
        raise ValueError("Portfolio slate generation returned no strategies")
    return slate.portfolios[:slate_size]


def _format_prompt(user_request: str, slate_size: int) -> str:
    return (
        f'Original portfolio research idea: "{user_request}"\n\n'
        f"Produce a pre-registered slate of {slate_size} distinct PortfolioSpec entries."
    )
