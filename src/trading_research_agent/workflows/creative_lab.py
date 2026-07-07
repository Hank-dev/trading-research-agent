"""Creative-but-disciplined portfolio strategy lab.

This workflow deliberately separates creativity from validation:

1. Generate a finite, deterministic, pre-registered slate of structurally distinct
   portfolio rules from a small idea palette.
2. Run every candidate on the train segment exactly once.
3. Re-test every train survivor on a held-out lockbox.
4. Stress-test only lockbox survivors with existing robustness perturbations.
5. Report denominators at every gate. No parameter mutation is allowed after seeing
   results, so the loop can be creative without becoming an overfit machine.
"""
from __future__ import annotations

from typing import Any, TypedDict

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.data_loader import load_portfolio_panel
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.workflows.anomaly_miner import mine_anomalies
from trading_research_agent.workflows.portfolio_research import run_portfolio_backtest
from trading_research_agent.workflows.researched_hypothesis_generator import (
    generate_researched_portfolio_slate,
)
from trading_research_agent.workflows.robustness_stress import run_stress_test


class CreativeLabResult(TypedDict, total=False):
    assets: list[str]
    full_start: str
    full_end: str
    lockbox_pct: float
    train_start: str
    train_end: str
    lockbox_start: str
    slate: list[PortfolioSpec]
    train: list[dict[str, Any]]
    lockbox: list[dict[str, Any]]
    stress: list[dict[str, Any]]
    winner: dict[str, Any] | None
    summary: dict[str, Any]
    research_brief: str
    hypotheses: list[Any]
    anomaly_facts: list[str]
    errors: list[str]


def generate_creative_slate(
    assets: list[str],
    start: str,
    end: str,
    max_candidates: int = 8,
) -> list[PortfolioSpec]:
    """Build a bounded, deterministic slate of structurally different rules.

    The palette is intentionally small and hand-curated. It is not a parameter
    optimizer: each idea gets at most one or two broad, conventional settings.
    """
    clean_assets = _clean_assets(assets)
    if len(clean_assets) < 2:
        raise ValueError("Creative lab needs at least 2 assets")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    top_k_1 = 1
    top_k_diversified = min(2, len(clean_assets))
    long_lookback = 252
    short_lookback = 63
    medium_lookback = 126

    ideas: list[dict[str, Any]] = [
        {
            "family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
            "lookback_days": medium_lookback,
            "top_k": top_k_diversified,
            "rebalance_days": 21,
            "hypothesis": "Cross-asset momentum: capital rotates toward the strongest recent winners.",
        },
        {
            "family": PortfolioFamily.DUAL_MOMENTUM,
            "lookback_days": long_lookback,
            "top_k": top_k_1,
            "rebalance_days": 21,
            "hypothesis": "Dual momentum: only hold relative winners whose own trend is positive; otherwise de-risk.",
        },
        {
            "family": PortfolioFamily.EQUAL_WEIGHT_TREND,
            "lookback_days": long_lookback,
            "top_k": top_k_diversified,
            "rebalance_days": 21,
            "hypothesis": "Equal-weight trend filter: diversify, but switch off assets below their trend.",
        },
        {
            "family": PortfolioFamily.TIME_SERIES_MOMENTUM,
            "lookback_days": medium_lookback,
            "top_k": top_k_diversified,
            "rebalance_days": 21,
            "hypothesis": "Time-series momentum: each asset earns its place independently when its own trailing return is positive.",
        },
        {
            "family": PortfolioFamily.VOLATILITY_SCALED_MOMENTUM,
            "lookback_days": medium_lookback,
            "top_k": top_k_diversified,
            "rebalance_days": 21,
            "hypothesis": "Vol-scaled momentum: preserve trend exposure while preventing high-vol assets from dominating risk.",
        },
        {
            "family": PortfolioFamily.CROSS_SECTIONAL_REVERSAL,
            "lookback_days": 756,
            "skip_recent_days": 126,
            "top_k": top_k_1,
            "rebalance_days": 21,
            "hypothesis": "Long-horizon reversal with a skip-recent gap: bet on multi-year losers without fighting short-term momentum.",
        },
        {
            "family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
            "lookback_days": short_lookback,
            "top_k": top_k_1,
            "rebalance_days": 5,
            "hypothesis": "Fast tactical momentum: short-window rotation should only survive if the effect is real, not a speed-picked accident.",
        },
        {
            "family": PortfolioFamily.DUAL_MOMENTUM,
            "lookback_days": medium_lookback,
            "top_k": top_k_diversified,
            "rebalance_days": 63,
            "hypothesis": "Slow dual momentum: lower turnover version should preserve structural edge if trend is genuine.",
        },
    ]

    slate: list[PortfolioSpec] = []
    for idx, idea in enumerate(ideas, start=1):
        if len(slate) >= max_candidates:
            break
        data = {
            "name": f"Creative {idx}: {idea['family'].value}",
            "assets": clean_assets,
            "portfolio_family": idea["family"],
            "start_date": start,
            "end_date": end,
            "lookback_days": idea["lookback_days"],
            "top_k": idea.get("top_k", top_k_1),
            "rebalance_days": idea["rebalance_days"],
            "skip_recent_days": idea.get("skip_recent_days", 252),
            "hypothesis": idea["hypothesis"],
        }
        try:
            slate.append(PortfolioSpec(**data))
        except Exception:
            # Invalid templates are skipped rather than repaired from results.
            continue
    return slate


def run_creative_lab(
    assets: list[str],
    start: str,
    end: str,
    max_candidates: int = 8,
    lockbox_pct: float = 0.25,
    research_goal: str | None = None,
) -> CreativeLabResult:
    if not 0.0 < lockbox_pct < 1.0:
        raise ValueError("lockbox_pct must be in (0, 1)")

    full_assets = _clean_assets(assets)
    train_end, lockbox_start = split_date_range(start, end, lockbox_pct)
    research_brief = ""
    hypotheses: list[Any] = []
    anomaly_facts: list[str] = []
    if research_goal:
        anomaly_result = mine_anomalies(full_assets, start, end, top_n=max_candidates * 3)
        anomaly_facts = _format_anomaly_facts_for_research(
            anomaly_result.get("facts", [])
        )
        researched = generate_researched_portfolio_slate(
            user_request=research_goal,
            assets=full_assets,
            start=start,
            end=train_end,
            slate_size=max_candidates,
            anomaly_facts=anomaly_facts,
        )
        slate = researched.portfolios
        research_brief = researched.research_brief
        hypotheses = list(researched.hypotheses)
    else:
        slate = generate_creative_slate(full_assets, start, train_end, max_candidates)
    panel = load_portfolio_panel(full_assets, start, end)

    train_results: list[dict[str, Any]] = []
    lockbox_results: list[dict[str, Any]] = []
    stress_results: list[dict[str, Any]] = []

    for spec in slate:
        train_results.append(
            run_portfolio_backtest(spec, "creative-lab train: pre-registered candidate", panel=panel)
        )

    train_survivors = [
        state["strategy_spec"]
        for state in train_results
        if _verdict_of(state) == "worth_paper_trading"
    ]

    for spec in train_survivors:
        lockbox_spec = spec.model_copy(
            update={
                "name": f"{spec.name} (lockbox)",
                "start_date": lockbox_start,
                "end_date": end,
            }
        )
        lockbox_results.append(
            run_portfolio_backtest(
                lockbox_spec,
                "creative-lab held-out lockbox: no mutation after train screen",
                panel=panel,
            )
        )

    lockbox_survivors = [
        state["strategy_spec"]
        for state in lockbox_results
        if _verdict_of(state) == "worth_paper_trading"
    ]

    for lockbox_spec in lockbox_survivors:
        full_spec = lockbox_spec.model_copy(
            update={
                "name": lockbox_spec.name.replace(" (lockbox)", ""),
                "start_date": start,
                "end_date": end,
            }
        )
        stress = run_stress_test(full_spec, start, end, panel=panel)
        stress_results.append({"strategy_spec": full_spec, "stress": stress})

    robust_survivors = [
        item for item in stress_results if item["stress"].get("summary", {}).get("verdict") == "ROBUST"
    ]
    winner = _select_winner(robust_survivors)
    summary = _summarize(
        slate_count=len(slate),
        train_count=len(train_survivors),
        lockbox_count=len(lockbox_survivors),
        stress_count=len(robust_survivors),
        winner=winner,
    )

    return {
        "assets": full_assets,
        "full_start": start,
        "full_end": end,
        "lockbox_pct": lockbox_pct,
        "train_start": start,
        "train_end": train_end,
        "lockbox_start": lockbox_start,
        "slate": slate,
        "train": train_results,
        "lockbox": lockbox_results,
        "stress": stress_results,
        "winner": winner,
        "summary": summary,
        "research_brief": research_brief,
        "hypotheses": hypotheses,
        "anomaly_facts": anomaly_facts,
    }


def _format_anomaly_facts_for_research(facts: list[Any]) -> list[str]:
    lines: list[str] = []
    for fact in facts:
        fact_text = getattr(fact, "fact", "")
        control = getattr(fact, "control", "")
        if fact_text:
            lines.append(f"{fact_text} Control: {control}" if control else fact_text)
    return lines


def _clean_assets(assets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in assets:
        asset = raw.strip()
        if asset and asset.lower() not in seen:
            seen.add(asset.lower())
            out.append(asset)
    return out


def _verdict_of(state: dict[str, Any]) -> str | None:
    report = state.get("report")
    return report.verdict if report is not None else None


def _select_winner(robust_survivors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not robust_survivors:
        return None

    def score(item: dict[str, Any]) -> tuple[float, float, str]:
        stress = item["stress"]
        summary = stress.get("summary", {})
        return (
            float(summary.get("overall_rate", 0.0)),
            float(summary.get("overall_confirmed", 0)),
            item["strategy_spec"].name,
        )

    return max(robust_survivors, key=score)


def _summarize(
    slate_count: int,
    train_count: int,
    lockbox_count: int,
    stress_count: int,
    winner: dict[str, Any] | None,
) -> dict[str, Any]:
    if slate_count == 0:
        verdict = "REJECTED_NO_RUNNABLE_CANDIDATES"
    elif train_count == 0:
        verdict = "REJECTED_NO_TRAIN_SURVIVORS"
    elif lockbox_count == 0:
        verdict = "REJECTED_NO_LOCKBOX_SURVIVORS"
    elif stress_count == 0:
        verdict = "REJECTED_FRAGILE_LOCKBOX_SURVIVORS"
    elif winner is not None:
        verdict = "PAPER_TRADE_CANDIDATE"
    else:
        verdict = "REJECTED_UNKNOWN"

    return {
        "pre_registered_candidates": slate_count,
        "train_survivors": train_count,
        "lockbox_survivors": lockbox_count,
        "stress_survivors": stress_count,
        "verdict": verdict,
        "anti_overfit_rule": (
            "Finite pre-registered slate; no candidate is mutated after seeing results; "
            "only held-out lockbox and robustness survival can promote a strategy."
        ),
    }
