"""Optional Grok interpretation of a robustness stress-test result.

Reads the deterministic survival numbers and gives an honest plain-English read:
is the edge robust or a knife-edge, and what should the researcher do. It never
changes the numbers — the deterministic verdict in robustness_stress is
authoritative; this only narrates it.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from trading_research_agent.config import load_settings


SYSTEM_PROMPT = """You are interpreting a robustness stress-test of a trading strategy that already passed a held-out lockbox once.

The stress test perturbed the strategy and re-checked, for each perturbation, whether it STILL confirms on a held-out segment:
- lockbox: same strategy, different held-out cut points (0.15/0.20/0.25/0.30 of the range).
- parameter: nearby lookback / rebalance / top_k values.
- universe: leave-one-out (drop each asset in turn).

Your job is to be the honest skeptic:
- If most perturbations still confirm, the edge is robust and the original pass is more believable.
- If only the exact original settings confirm and neighbors collapse, it is a knife-edge / likely overfit, no matter how good the headline pass looked.
- If dropping ONE specific asset destroys the edge, the strategy is really a bet on that asset, not a portfolio effect — say so and name the asset.
- If different lockbox cut points disagree, the original lockbox pass may have been a lucky window.

Be concrete and refer to the actual survival rates. Do not suggest parameter tweaks to "fix" a fragile strategy — that is the overfitting trap. If it is fragile, the honest recommendation is to distrust it, not to tune it. Recommend paper trading (forward out-of-sample) over any further in-sample work. Never claim profitability.
"""


class RobustnessInterpretation(BaseModel):
    assessment: str = Field(description="2-4 sentence honest read of the fragility pattern.")
    fragility_flags: list[str] = Field(
        description="Specific weaknesses found (e.g. 'collapses when GLD is dropped')."
    )
    recommendation: str = Field(
        description="What to do next. Favor paper trading; never suggest tuning a fragile strategy."
    )


def interpret_robustness(stress_result: dict) -> RobustnessInterpretation:
    settings = load_settings()
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.model,
        temperature=0,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    structured_model = model.with_structured_output(RobustnessInterpretation)
    parsed = structured_model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=_format_prompt(stress_result)),
        ]
    )
    if isinstance(parsed, RobustnessInterpretation):
        return parsed
    return RobustnessInterpretation.model_validate(parsed)


def _format_prompt(stress_result: dict) -> str:
    summary = stress_result.get("summary", {})
    lines = [
        f"Strategy: {stress_result.get('spec_name')}",
        f"Universe: {stress_result.get('universe')}",
        f"Date range: {stress_result.get('full_start')} to {stress_result.get('full_end')}",
        f"Deterministic verdict: {summary.get('verdict')}",
        f"Overall held-out survival: {summary.get('overall_confirmed')}/{summary.get('overall_runnable')} "
        f"({summary.get('overall_rate', 0):.0%})",
        "",
        "Survival by category:",
    ]
    for category, stats in summary.get("category_rates", {}).items():
        lines.append(f"  {category}: {stats['confirmed']}/{stats['total']} ({stats['rate']:.0%})")

    lines.append("")
    lines.append("Per-perturbation detail (held-out segment):")
    for r in stress_result.get("results", []):
        if r.get("status") != "ok":
            lines.append(f"  [{r.get('category')}] {r.get('label')}: UNRUNNABLE ({r.get('detail')})")
            continue
        mark = "CONFIRMS" if r.get("confirms") else "fails"
        ret = r.get("held_out_return_pct")
        bench = r.get("held_out_benchmark_pct")
        sharpe = r.get("held_out_sharpe")
        ret_s = f"{ret:.1f}%" if ret is not None else "n/a"
        bench_s = f"{bench:.1f}%" if bench is not None else "n/a"
        sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "n/a"
        lines.append(
            f"  [{r.get('category')}] {r.get('label')}: {mark} "
            f"(held-out {ret_s} vs benchmark {bench_s}, Sharpe {sharpe_s})"
        )

    lines.append("")
    lines.append("Give an honest assessment, fragility flags, and a recommendation.")
    return "\n".join(lines)
