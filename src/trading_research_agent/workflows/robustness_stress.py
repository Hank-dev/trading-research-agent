"""Robustness stress-testing for a confirmed portfolio winner.

The philosophy is the opposite of optimization: we deliberately PERTURB a winning
strategy and measure how much of its edge survives. A real edge is robust to its
own neighborhood — nearby parameters, different held-out windows, and dropping any
single asset. A lucky overfit is a knife-edge that collapses the moment you nudge it.

For each perturbation we re-run the full train/lockbox split and ask the only
question that matters: does it still confirm on the held-out segment? We report
survival rates per category and an overall verdict of ROBUST / FRAGILE / BROKEN.

This module is deterministic. The optional Grok interpretation lives in
nodes.interpret_robustness and only reads these numbers; it never drives them.
"""

from collections import defaultdict
from typing import Any, TypedDict

import pandas as pd

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.config import DEFAULT_LOCKBOX_PCT
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.workflows.portfolio_research import run_portfolio_backtest

DEFAULT_LOCKBOX_GRID = (0.15, 0.20, 0.25, 0.30)
_OVERALL_ROBUST_RATE = 0.70
_CATEGORY_MIN_RATE = 0.50


class PerturbationResult(TypedDict, total=False):
    category: str
    label: str
    status: str  # "ok" | "unrunnable"
    confirms: bool
    held_out_verdict: str | None
    train_verdict: str | None
    held_out_return_pct: float | None
    held_out_benchmark_pct: float | None
    held_out_sharpe: float | None
    detail: str


class StressResult(TypedDict, total=False):
    spec_name: str
    universe: list[str]
    full_start: str
    full_end: str
    base_lockbox_pct: float
    results: list[PerturbationResult]
    summary: dict[str, Any]


def run_stress_test(
    spec: PortfolioSpec,
    full_start: str,
    full_end: str,
    base_lockbox_pct: float = DEFAULT_LOCKBOX_PCT,
    lockbox_grid: tuple[float, ...] = DEFAULT_LOCKBOX_GRID,
    panel: pd.DataFrame | None = None,
) -> StressResult:
    """Stress-test a confirmed winner across lockbox cuts, parameter neighbors,
    and leave-one-out universes. `panel` may be a pre-loaded master price panel
    spanning the full universe and date range (avoids repeated network loads)."""
    if panel is None:
        from trading_research_agent.tools.data_loader import load_portfolio_panel

        panel = load_portfolio_panel(spec.assets, full_start, full_end)

    results: list[PerturbationResult] = []

    # 1. Lockbox sensitivity: same strategy, different held-out cut points.
    grid = tuple(sorted({base_lockbox_pct, *lockbox_grid}))
    for pct in grid:
        ev = _evaluate(spec, full_start, full_end, pct, panel)
        ev["category"] = "lockbox"
        ev["label"] = f"lockbox_pct={pct:.2f}"
        results.append(ev)

    # 2. Parameter neighbors and 3. leave-one-out universe (at the base cut).
    for category, label, variant in _build_perturbations(spec):
        ev = _evaluate(variant, full_start, full_end, base_lockbox_pct, panel)
        ev["category"] = category
        ev["label"] = label
        results.append(ev)

    summary = _summarize(results, base_lockbox_pct)
    return {
        "spec_name": spec.name,
        "universe": list(spec.assets),
        "full_start": full_start,
        "full_end": full_end,
        "base_lockbox_pct": base_lockbox_pct,
        "results": results,
        "summary": summary,
    }


def _evaluate(
    spec: PortfolioSpec,
    full_start: str,
    full_end: str,
    lockbox_pct: float,
    panel: pd.DataFrame,
) -> PerturbationResult:
    try:
        train_end, lockbox_start = split_date_range(full_start, full_end, lockbox_pct)
    except ValueError as exc:
        return {"status": "unrunnable", "confirms": False, "detail": str(exc)}

    train_spec = spec.model_copy(update={"start_date": full_start, "end_date": train_end})
    lockbox_spec = spec.model_copy(
        update={
            "start_date": lockbox_start,
            "end_date": full_end,
            "name": f"{spec.name} (lockbox)",
        }
    )

    train_state = run_portfolio_backtest(train_spec, "stress: train", panel=panel)
    lockbox_state = run_portfolio_backtest(lockbox_spec, "stress: held-out", panel=panel)

    train_verdict = _verdict_of(train_state)
    held_out_verdict = _verdict_of(lockbox_state)
    held = lockbox_state.get("backtest_result")

    if held is None:
        return {
            "status": "unrunnable",
            "confirms": False,
            "train_verdict": train_verdict,
            "held_out_verdict": held_out_verdict,
            "detail": "; ".join(lockbox_state.get("errors", [])) or "no held-out backtest",
        }

    m = held.metrics
    return {
        "status": "ok",
        "confirms": held_out_verdict == "worth_paper_trading",
        "train_verdict": train_verdict,
        "held_out_verdict": held_out_verdict,
        "held_out_return_pct": m.total_return_pct,
        "held_out_benchmark_pct": m.buy_and_hold_return_pct,
        "held_out_sharpe": m.sharpe_ratio,
        "detail": "",
    }


def _build_perturbations(spec: PortfolioSpec) -> list[tuple[str, str, PortfolioSpec]]:
    perturbations: list[tuple[str, str, PortfolioSpec]] = []

    # Parameter neighbors: vary one knob at a time around the base.
    for lookback in sorted({int(spec.lookback_days * 0.8), int(spec.lookback_days * 1.25)}):
        variant = _make_variant(spec, lookback_days=lookback)
        if variant is not None and variant.lookback_days != spec.lookback_days:
            perturbations.append(("parameter", f"lookback={variant.lookback_days}", variant))

    for rebalance in sorted({max(1, int(spec.rebalance_days * 0.5)), spec.rebalance_days * 2}):
        variant = _make_variant(spec, rebalance_days=rebalance)
        if variant is not None and variant.rebalance_days != spec.rebalance_days:
            perturbations.append(("parameter", f"rebalance={variant.rebalance_days}", variant))

    if spec.portfolio_family in (
        PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        PortfolioFamily.DUAL_MOMENTUM,
    ):
        for top_k in {spec.top_k - 1, spec.top_k + 1}:
            variant = _make_variant(spec, top_k=top_k)
            if variant is not None and variant.top_k != spec.top_k:
                perturbations.append(("parameter", f"top_k={variant.top_k}", variant))

    # Leave-one-out universe: does the edge depend on any single asset?
    if len(spec.assets) >= 3:
        for dropped in spec.assets:
            subset = [a for a in spec.assets if a != dropped]
            variant = _make_variant(
                spec, assets=subset, top_k=min(spec.top_k, len(subset))
            )
            if variant is not None:
                perturbations.append(("universe", f"drop {dropped}", variant))

    return perturbations


def _make_variant(spec: PortfolioSpec, **updates: Any) -> PortfolioSpec | None:
    data = spec.model_dump()
    data.update(updates)
    try:
        return PortfolioSpec(**data)
    except Exception:
        return None


def _summarize(results: list[PerturbationResult], base_lockbox_pct: float) -> dict[str, Any]:
    runnable = [r for r in results if r.get("status") == "ok"]
    by_category: dict[str, list[PerturbationResult]] = defaultdict(list)
    for r in runnable:
        by_category[r["category"]].append(r)

    category_rates: dict[str, dict[str, Any]] = {}
    for category, items in by_category.items():
        confirmed = sum(1 for r in items if r["confirms"])
        category_rates[category] = {
            "confirmed": confirmed,
            "total": len(items),
            "rate": confirmed / len(items) if items else 0.0,
        }

    confirmed_total = sum(1 for r in runnable if r["confirms"])
    overall_rate = confirmed_total / len(runnable) if runnable else 0.0

    base = next(
        (
            r
            for r in results
            if r.get("category") == "lockbox"
            and r.get("label") == f"lockbox_pct={base_lockbox_pct:.2f}"
        ),
        None,
    )
    base_confirms = bool(base and base.get("status") == "ok" and base.get("confirms"))

    if not base_confirms:
        verdict = "BROKEN"
    elif (
        overall_rate >= _OVERALL_ROBUST_RATE
        and category_rates
        and min(c["rate"] for c in category_rates.values()) >= _CATEGORY_MIN_RATE
    ):
        verdict = "ROBUST"
    else:
        verdict = "FRAGILE"

    return {
        "verdict": verdict,
        "base_confirms": base_confirms,
        "overall_confirmed": confirmed_total,
        "overall_runnable": len(runnable),
        "overall_rate": overall_rate,
        "category_rates": category_rates,
        "unrunnable": sum(1 for r in results if r.get("status") != "ok"),
    }


def _verdict_of(state: dict[str, Any]) -> str | None:
    report = state.get("report")
    return report.verdict if report is not None else None


def latest_confirmed_portfolio_winner(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reconstruct the most recent lockbox-confirmed portfolio winner from history.

    Joins the in-slate and lockbox records by slate_id to recover the full date
    range. Returns a dict with the spec fields and full range, or None."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("mode") == "portfolio" and r.get("slate_id"):
            groups[r["slate_id"]].append(r)

    best: dict[str, Any] | None = None
    best_ts = ""
    for recs in groups.values():
        confirmed_lockbox = [
            r
            for r in recs
            if r.get("is_lockbox") and r.get("verdict") == "worth_paper_trading"
        ]
        if not confirmed_lockbox:
            continue
        winner = confirmed_lockbox[0]
        params = winner.get("params", {})
        if "assets" not in params:
            continue
        starts = [r["start_date"] for r in recs if "start_date" in r]
        ends = [r["end_date"] for r in recs if "end_date" in r]
        timestamps = [r.get("timestamp", "") for r in recs]
        candidate = {
            "strategy_family": winner.get("strategy_family"),
            "params": params,
            "full_start": min(starts) if starts else None,
            "full_end": max(ends) if ends else None,
            "timestamp": max(timestamps) if timestamps else "",
        }
        if candidate["full_start"] and candidate["full_end"] and candidate["timestamp"] > best_ts:
            best = candidate
            best_ts = candidate["timestamp"]

    return best


def spec_from_winner(winner: dict[str, Any]) -> PortfolioSpec:
    params = winner["params"]
    return PortfolioSpec(
        name=f"Stress: {winner['strategy_family']} {params['assets']}",
        assets=params["assets"],
        portfolio_family=PortfolioFamily(winner["strategy_family"]),
        start_date=winner["full_start"],
        end_date=winner["full_end"],
        lookback_days=params.get("lookback_days", 126),
        top_k=params.get("top_k", 1),
        rebalance_days=params.get("rebalance_days", 21),
        hypothesis="Reconstructed confirmed winner under robustness stress test.",
    )
