"""Parameter sweep as a robustness test, not a peak-picker.

Sweeping a parameter (e.g. Donchian entry_window) over a range is the highest
overfitting-risk activity in quant research — IF you crown the single best value.
But the same runs, read as a surface, are evidence about robustness:

- A contiguous PLATEAU of values that all confirm on the held-out lockbox means
  the edge tolerates the knob being wrong — a real phenomenon, not a magic number.
- An isolated SPIKE (one value confirms, its neighbors fail) is the signature of
  curve-fitting, no matter how good that one value looks.

So this never reports "the best is X." It pre-registers the whole range, runs
every value through the same train/lockbox split, and asks whether the confirmers
form a plateau or a spike. Adjacent parameter values produce highly correlated
signals, so these are NOT independent trials — the deflated Sharpe across the
sweep is reported with that caveat.
"""

from typing import Any, TypedDict

from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.tools.stats import (
    deflated_sharpe_ratio,
    estimate_trading_days,
)
from trading_research_agent.workflows.research_graph import build_research_graph

_PLATEAU_MIN_RUN = 3  # >= 3 contiguous confirming values = a robust plateau

_FAMILY_DEFAULTS: dict[StrategyFamily, dict[str, Any]] = {
    StrategyFamily.SMA_CROSSOVER: {"fast_window": 50, "slow_window": 200},
    StrategyFamily.DONCHIAN_BREAKOUT: {"entry_window": 55, "exit_window": 20},
    StrategyFamily.RSI_MEAN_REVERSION: {
        "rsi_window": 14,
        "oversold_threshold": 30.0,
        "exit_threshold": 50.0,
    },
    StrategyFamily.FILTERED_DONCHIAN_BREAKOUT: {
        "entry_window": 55,
        "exit_window": 20,
        "atr_window": 14,
        "atr_ma_window": 20,
        "regime_window": 200,
    },
}

_SWEEPABLE: dict[StrategyFamily, set[str]] = {
    StrategyFamily.SMA_CROSSOVER: {"fast_window", "slow_window"},
    StrategyFamily.DONCHIAN_BREAKOUT: {"entry_window", "exit_window"},
    StrategyFamily.RSI_MEAN_REVERSION: {"rsi_window", "oversold_threshold", "exit_threshold"},
    StrategyFamily.FILTERED_DONCHIAN_BREAKOUT: {
        "entry_window",
        "exit_window",
        "atr_window",
        "atr_ma_window",
        "regime_window",
    },
}

_INT_PARAMS = {
    "fast_window",
    "slow_window",
    "entry_window",
    "exit_window",
    "rsi_window",
    "atr_window",
    "atr_ma_window",
    "regime_window",
}


class SweepEntry(TypedDict, total=False):
    value: float
    status: str  # "ok" | "invalid"
    train_verdict: str | None
    lockbox_verdict: str | None
    confirms: bool
    held_out_sharpe: float | None
    held_out_return_pct: float | None
    held_out_benchmark_pct: float | None
    detail: str


class SweepResult(TypedDict, total=False):
    asset: str
    family: str
    param: str
    values: list[float]
    lockbox_pct: float
    entries: list[SweepEntry]
    plateau_values: list[float]
    longest_run: int
    n_confirmed: int
    sweep_dsr: float | None
    verdict: str


def run_single_asset_sweep(
    asset: str,
    family: StrategyFamily,
    param: str,
    values: list[float],
    start: str,
    end: str,
    lockbox_pct: float,
) -> SweepResult:
    if param not in _SWEEPABLE.get(family, set()):
        raise ValueError(
            f"{param} is not sweepable for {family.value}; "
            f"choose one of {sorted(_SWEEPABLE.get(family, set()))}"
        )
    if lockbox_pct <= 0:
        raise ValueError("parameter sweep requires --lockbox-pct > 0")
    if len(values) < 2:
        raise ValueError("sweep needs at least 2 values")

    train_end, lockbox_start = split_date_range(start, end, lockbox_pct)
    graph = build_research_graph()

    entries: list[SweepEntry] = []
    for raw in values:
        value = int(round(raw)) if param in _INT_PARAMS else float(raw)
        entries.append(
            _evaluate_value(graph, asset, family, param, value, start, end, train_end, lockbox_start)
        )

    entries.sort(key=lambda e: e["value"])
    longest_run, plateau_values = _longest_confirm_run(entries)
    n_confirmed = sum(1 for e in entries if e.get("confirms"))
    sweep_dsr = _sweep_dsr(entries, start, end, lockbox_pct)
    verdict = _verdict(n_confirmed, longest_run)

    return {
        "asset": asset,
        "family": family.value,
        "param": param,
        "values": [e["value"] for e in entries],
        "lockbox_pct": lockbox_pct,
        "entries": entries,
        "plateau_values": plateau_values,
        "longest_run": longest_run,
        "n_confirmed": n_confirmed,
        "sweep_dsr": sweep_dsr,
        "verdict": verdict,
    }


def _evaluate_value(
    graph, asset, family, param, value, full_start, full_end, train_end, lockbox_start
) -> SweepEntry:
    try:
        train_spec = _build_spec(asset, family, param, value, full_start, train_end)
        lockbox_spec = _build_spec(asset, family, param, value, lockbox_start, full_end)
    except Exception as exc:
        return {"value": value, "status": "invalid", "confirms": False, "detail": str(exc)}

    train_verdict, _ = _run_spec(graph, train_spec)
    lockbox_verdict, lockbox_metrics = _run_spec(graph, lockbox_spec)

    entry: SweepEntry = {
        "value": value,
        "status": "ok",
        "train_verdict": train_verdict,
        "lockbox_verdict": lockbox_verdict,
        "confirms": lockbox_verdict == "worth_paper_trading",
        "detail": "",
    }
    if lockbox_metrics is not None:
        entry["held_out_sharpe"] = lockbox_metrics.sharpe_ratio
        entry["held_out_return_pct"] = lockbox_metrics.total_return_pct
        entry["held_out_benchmark_pct"] = lockbox_metrics.buy_and_hold_return_pct
    return entry


def _build_spec(asset, family, param, value, start, end) -> StrategySpec:
    fields = dict(_FAMILY_DEFAULTS[family])
    fields[param] = value
    return StrategySpec(
        name=f"{family.value} {param}={value}",
        asset=asset,
        strategy_family=family,
        start_date=start,
        end_date=end,
        hypothesis=f"Sweep variant {param}={value} for {family.value} on {asset}.",
        **fields,
    )


def _run_spec(graph, spec: StrategySpec) -> tuple[str | None, Any]:
    state = graph.invoke({"strategy_spec": spec})
    report = state.get("report")
    result = state.get("backtest_result")
    verdict = report.verdict if report is not None else None
    metrics = result.metrics if result is not None else None
    return verdict, metrics


def _longest_confirm_run(entries_sorted: list[SweepEntry]) -> tuple[int, list[float]]:
    best, current = 0, 0
    best_vals: list[float] = []
    current_vals: list[float] = []
    for e in entries_sorted:
        if e.get("status") == "ok" and e.get("confirms"):
            current += 1
            current_vals.append(e["value"])
            if current > best:
                best, best_vals = current, list(current_vals)
        else:
            current = 0
            current_vals = []
    return best, best_vals


def _sweep_dsr(entries: list[SweepEntry], start: str, end: str, lockbox_pct: float) -> float | None:
    sharpes = [
        e["held_out_sharpe"]
        for e in entries
        if e.get("status") == "ok" and e.get("held_out_sharpe") is not None
    ]
    if len(sharpes) < 2:
        return None
    confirming = [
        e["held_out_sharpe"]
        for e in entries
        if e.get("confirms") and e.get("held_out_sharpe") is not None
    ]
    if not confirming:
        return None
    best = max(confirming)
    _, lockbox_start = split_date_range(start, end, lockbox_pct)
    n_obs = estimate_trading_days(lockbox_start, end)
    return deflated_sharpe_ratio(best, n_obs, sharpes)


def _verdict(n_confirmed: int, longest_run: int) -> str:
    if n_confirmed == 0:
        return "NONE"
    if longest_run >= _PLATEAU_MIN_RUN:
        return "PLATEAU"
    return "SPIKE"


def format_sweep(result: SweepResult) -> str:
    lines = [
        f"Asset:   {result['asset']}",
        f"Family:  {result['family']}",
        f"Sweep:   {result['param']} over {result['values']}",
        f"Lockbox: {result['lockbox_pct']:.0%} held out",
        "",
        "Per-value held-out (lockbox) result:",
    ]
    for e in result["entries"]:
        if e.get("status") != "ok":
            lines.append(f"  {result['param']}={e['value']:<6} INVALID ({e.get('detail', '')})")
            continue
        mark = "CONFIRMS" if e.get("confirms") else "fails   "
        ret = e.get("held_out_return_pct")
        bench = e.get("held_out_benchmark_pct")
        ret_s = f"{ret:7.1f}%" if ret is not None else "    n/a"
        bench_s = f"{bench:7.1f}%" if bench is not None else "    n/a"
        lines.append(
            f"  {result['param']}={e['value']:<6} {mark}  held-out {ret_s} vs bench {bench_s}"
        )

    lines.append("")
    if result["plateau_values"]:
        lines.append(
            f"Longest contiguous confirming band: {result['plateau_values']} "
            f"({result['longest_run']} values)"
        )
    if result.get("sweep_dsr") is not None:
        lines.append(
            f"Deflated Sharpe of best confirmer across the sweep: {result['sweep_dsr']:.3f} "
            "(NOTE: adjacent values are correlated, so this overstates independence)"
        )
    lines.append("")
    lines.append(f"VERDICT: {result['verdict']}")
    lines.append("")
    lines.append(_verdict_gloss(result["verdict"]))
    return "\n".join(lines)


def _verdict_gloss(verdict: str) -> str:
    if verdict == "PLATEAU":
        return (
            "A contiguous band of parameter values all confirm out of sample. The edge "
            "tolerates the knob being wrong, so it is a robust phenomenon rather than a "
            "value curve-fit to history. Do NOT pick the single best value to trade — "
            "pick the middle of the plateau, and forward paper-trade it."
        )
    if verdict == "SPIKE":
        return (
            "Only isolated value(s) confirm while their neighbors fail. This is the "
            "classic overfitting signature: the result depends on one magic number. "
            "Distrust it — a real edge does not vanish when you nudge the parameter."
        )
    return (
        "No parameter value confirmed on the held-out lockbox. There is no edge here to "
        "tune. That is a clean finding; stop sweeping this strategy."
    )
