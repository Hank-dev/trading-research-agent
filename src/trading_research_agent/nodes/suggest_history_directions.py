"""LLM-powered analysis of cross-run history.

Reads the deterministic summary built by `tools.history` and asks Grok (via the
existing config) for structural observations and at-most-three next directions.

This is deliberately scoped to *structural* suggestions — different asset
classes, different families, different regimes, stopping entirely — rather than
parameter tweaks of strategies already tried. Parameter-tweak suggestions are
the multiple-testing trap that the rest of the pipeline pushes back against;
having the suggestion layer reintroduce it would defeat the design.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trading_research_agent.config import load_settings


SYSTEM_PROMPT = """You are reading a researcher's cross-session backtest log to help them see patterns they may have missed.

You are NOT to suggest parameter tweaks of strategies they have already tried — that would compound the multiple-testing problem that the rest of this pipeline is designed to push back against. Examples of forbidden suggestions: "try fast_window=55 slow_window=210", "try RSI(13) instead of RSI(14)", "extend the date range by one year".

Your job:
1. Identify STRUCTURAL gaps — what asset classes, strategy families, or regimes have NOT been tried.
2. Flag if their failure patterns suggest a methodological issue. For example:
   - Always failing "Benchmark comparison" means their strategies reduce risk but don't increase absolute return. That's the expected signature of trend-following and mean-reversion in trending markets; the benchmark gate may be the wrong question for their research.
   - Always failing "Deflated Sharpe ratio (DSR)" with many trials means multiple-testing inflation is the dominant problem.
   - Always failing PSR with low trade counts means sample size is too small.
3. Be willing to recommend STOPPING the search and changing direction if the data supports it. "Stop testing simple rules on liquid public markets, build richer signals instead" is a valid recommendation. So is "the bar you've set may be wrong for the kind of strategy you're really trying to discover."
4. Acknowledge the cumulative trial count. The more they have tried, the higher the bar for any one passing strategy to be real.
5. Limit `next_directions` to at most THREE concrete, STRUCTURAL items. Each must be something genuinely different — a new asset class, a new family, a new methodological angle — not a variation of something already tried.
6. Be honest. If the right answer is "your methodology is fine, you're just discovering that simple rules don't work on liquid markets," say so.

Do not mirror the user's own language. Do not invent metrics. Do not claim profitability.
"""


class HistorySuggestion(BaseModel):
    summary: str = Field(
        description="2-4 sentence honest assessment of what the history shows."
    )
    structural_gaps: list[str] = Field(
        description="Asset classes, strategy families, or regimes the researcher has not yet tried."
    )
    next_directions: list[str] = Field(
        max_length=3,
        description="At most 3 concrete STRUCTURAL next directions — not parameter tweaks.",
    )
    honest_warnings: list[str] = Field(
        description=(
            "Things to stop doing or to be skeptical about, including a recommendation to "
            "stop searching entirely if the data supports it."
        ),
    )


def suggest_directions_from_history(summary: dict) -> HistorySuggestion:
    settings = load_settings()
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.model,
        temperature=0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    structured_model = model.with_structured_output(HistorySuggestion)
    parsed = structured_model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=format_history_prompt(summary)),
        ]
    )
    if isinstance(parsed, HistorySuggestion):
        return parsed
    return HistorySuggestion.model_validate(parsed)


def format_history_prompt(summary: dict) -> str:
    sections = [
        f"Total trials across all sessions: {summary.get('total_trials', 0)}",
        f"Held-out lockbox verifications: {summary.get('lockbox_runs', 0)}",
        "",
        "Verdict distribution:",
        _bullets_from_counter(summary.get("by_verdict", {})),
        "",
        "Assets tried:",
        _bullets_from_counter(summary.get("by_asset", {})),
        "",
        "Strategy families tried:",
        _bullets_from_counter(summary.get("by_family", {})),
        "",
        "Asset/family pairs tried:",
        _bullets_from_counter(summary.get("asset_family_pairs", {})),
        "",
        "Most-failed robustness checks across trials:",
        _bullets_from_counter(summary.get("failed_checks", {})),
        "",
        f"Trials that reached worth_paper_trading: {len(summary.get('passed_runs', []))}",
    ]

    passed = summary.get("passed_runs", [])
    if passed:
        sections.append("")
        sections.append("Passing trials:")
        for r in passed:
            sections.append(
                f"  - {r.get('asset', '?')} {r.get('strategy_family', '?')} "
                f"({r.get('start_date', '?')} to {r.get('end_date', '?')})"
            )

    sections.append("")
    sections.append(
        "Produce structural observations and up to three structural next directions. "
        "If the honest answer is to stop searching, say so."
    )
    return "\n".join(sections)


def _bullets_from_counter(counter: dict) -> str:
    if not counter:
        return "  (none recorded)"
    return "\n".join(f"  - {count}x {key}" for key, count in counter.items())
