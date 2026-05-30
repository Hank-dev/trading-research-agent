"""Cumulative trial budget — honest multiple-testing accounting across the whole
research program.

The per-slate deflated Sharpe corrects for the trials inside one batch. But the
real multiple-testing surface is everything you have *ever* tried: every
portfolio backtest is a shot, and the bar a "winner" must clear rises with the
number of shots. This module counts the cumulative shots and re-deflates each
lockbox-confirmed candidate's out-of-sample Sharpe against that full count.

A strategy that cleared a single lockbox can still fail here — if you only found
it after testing fifty things, its edge has to be large enough to beat the best
of fifty coin-flips, not one.
"""

from typing import Any, TypedDict

from trading_research_agent.tools.stats import (
    deflated_sharpe_ratio,
    estimate_trading_days,
    probabilistic_sharpe_ratio,
)

_CROSS_RUN_PASS = 0.95


class CandidateAssessment(TypedDict):
    asset: str
    strategy_family: str
    held_out_sharpe: float
    cross_run_dsr: float
    clears_bar: bool
    start_date: str | None
    end_date: str | None


class TrialBudget(TypedDict):
    n_trials: int
    n_lockbox: int
    n_confirmed: int
    trial_sharpes: list[float]
    candidates: list[CandidateAssessment]
    best_cross_run_dsr: float | None
    verdict: str


def _is_portfolio(record: dict[str, Any]) -> bool:
    return str(record.get("asset", "")).startswith("PORTFOLIO[")


def _is_trial(record: dict[str, Any]) -> bool:
    # A "shot": a distinct portfolio hypothesis you evaluated. Lockbox re-tests are
    # confirmations, and stress perturbations probe one strategy's neighborhood —
    # neither is a new hypothesis you would have adopted, so they are excluded.
    return (
        _is_portfolio(record)
        and not record.get("is_lockbox", False)
        and record.get("mode") != "stress"
    )


def _sharpe(record: dict[str, Any]) -> float | None:
    metrics = record.get("metrics") or {}
    return metrics.get("sharpe_ratio")


def assess_trial_budget(records: list[dict[str, Any]]) -> TrialBudget:
    trials = [r for r in records if _is_trial(r)]
    trial_sharpes = [s for s in (_sharpe(r) for r in trials) if s is not None]

    lockboxes = [r for r in records if _is_portfolio(r) and r.get("is_lockbox")]
    confirmed = [r for r in lockboxes if r.get("verdict") == "worth_paper_trading"]

    n_trials = len(trials)
    candidates: list[CandidateAssessment] = []
    for r in confirmed:
        sharpe = _sharpe(r)
        if sharpe is None:
            continue
        start, end = r.get("start_date"), r.get("end_date")
        n_obs = estimate_trading_days(start, end) if start and end else 252
        if len(trial_sharpes) >= 2:
            dsr = deflated_sharpe_ratio(sharpe, n_obs, trial_sharpes)
        else:
            dsr = probabilistic_sharpe_ratio(sharpe, n_obs)
        candidates.append(
            {
                "asset": r.get("asset", "?"),
                "strategy_family": r.get("strategy_family", "?"),
                "held_out_sharpe": sharpe,
                "cross_run_dsr": dsr,
                "clears_bar": dsr >= _CROSS_RUN_PASS,
                "start_date": start,
                "end_date": end,
            }
        )

    best = max((c["cross_run_dsr"] for c in candidates), default=None)
    verdict = _verdict(n_trials, confirmed, candidates, best)

    return {
        "n_trials": n_trials,
        "n_lockbox": len(lockboxes),
        "n_confirmed": len(confirmed),
        "trial_sharpes": trial_sharpes,
        "candidates": candidates,
        "best_cross_run_dsr": best,
        "verdict": verdict,
    }


def _verdict(
    n_trials: int,
    confirmed: list[dict[str, Any]],
    candidates: list[CandidateAssessment],
    best: float | None,
) -> str:
    if n_trials == 0:
        return "NO_PORTFOLIO_RESEARCH_YET"
    if not confirmed:
        return "NOTHING_CONFIRMED"
    if best is not None and best >= _CROSS_RUN_PASS:
        return "SURVIVES_MULTIPLE_TESTING"
    return "CONFIRMED_BUT_NOT_AFTER_CORRECTION"


def format_budget(budget: TrialBudget) -> str:
    if budget["verdict"] == "NO_PORTFOLIO_RESEARCH_YET":
        return "No portfolio trials in history yet. Run --portfolio-spec or --portfolio."

    lines = [
        f"Portfolio shots taken (trials):     {budget['n_trials']}",
        f"Lockbox re-tests:                   {budget['n_lockbox']}",
        f"Lockbox-confirmed candidates:       {budget['n_confirmed']}",
        "",
        "Each shot spends statistical significance. A candidate must clear the "
        "deflated-Sharpe bar for the FULL shot count, not for a single slate.",
        "",
    ]

    if budget["candidates"]:
        lines.append("Lockbox-confirmed candidates, re-deflated against all shots:")
        for c in budget["candidates"]:
            mark = "CLEARS" if c["clears_bar"] else "fails "
            lines.append(
                f"  [{mark}] cross-run DSR={c['cross_run_dsr']:.3f}  "
                f"(held-out Sharpe {c['held_out_sharpe']:.2f})  "
                f"{c['strategy_family']} {c['asset']}"
            )
        lines.append("")

    lines.append(f"VERDICT: {budget['verdict']}")
    lines.append("")
    lines.append(_verdict_gloss(budget["verdict"]))
    return "\n".join(lines)


def _verdict_gloss(verdict: str) -> str:
    if verdict == "SURVIVES_MULTIPLE_TESTING":
        return (
            "A candidate's out-of-sample edge is large enough to clear the bar even "
            "after correcting for every strategy you have tried. This is the rarest, "
            "strongest result this toolkit can give. Forward paper-trade it; do not "
            "keep searching, which would only raise the bar again."
        )
    if verdict == "CONFIRMED_BUT_NOT_AFTER_CORRECTION":
        return (
            "Your candidate passed a single lockbox, but once corrected for how many "
            "strategies you tested, its edge is no longer distinguishable from the "
            "luckiest of that many draws. Treat it as unproven. More searching makes "
            "this worse, not better."
        )
    if verdict == "NOTHING_CONFIRMED":
        return (
            "No strategy has cleared a held-out lockbox yet. That is a legitimate "
            "finding: simple price rules on liquid assets rarely carry a durable edge. "
            "Each additional shot raises the bar for whatever you eventually find."
        )
    return ""
