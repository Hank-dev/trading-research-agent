"""Event-followthrough strategy generation and validation.

This fills the gap between anomaly facts and runnable strategies. It takes mined
`event_followthrough` facts, freezes a small slate of delayed event-trigger rules,
then evaluates train, lockbox, and parameter-neighborhood robustness without
mutating candidates after seeing results.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from trading_research_agent.tools.data_loader import load_portfolio_panel
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.workflows.anomaly_miner import AnomalyFact, mine_anomalies

Direction = Literal["strength", "weakness"]


@dataclass(frozen=True)
class EventFollowthroughSpec:
    name: str
    leader: str
    target: str
    direction: Direction
    z_threshold: float
    lag_days: int
    hold_days: int
    event_window: int = 20
    z_lookback_days: int = 252
    cost_per_turnover: float = 0.001
    hypothesis: str = ""


@dataclass(frozen=True)
class EventFollowthroughMetrics:
    total_return_pct: float
    annualized_return_pct: float
    sharpe: float | None
    max_drawdown_pct: float
    exposure_pct: float
    num_entries: int
    buy_hold_return_pct: float
    buy_hold_annualized_pct: float
    buy_hold_sharpe: float | None
    buy_hold_max_drawdown_pct: float
    beats_buy_hold_sharpe: bool


def generate_event_followthrough_slate(
    assets: list[str],
    start: str,
    end: str,
    *,
    max_candidates: int = 6,
    top_anomalies: int | None = None,
) -> list[EventFollowthroughSpec]:
    """Mine anomalies and freeze top event-followthrough facts as strategies."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    anomaly_result = mine_anomalies(
        assets,
        start,
        end,
        top_n=top_anomalies or max(max_candidates * 4, 12),
    )
    facts = [fact for fact in anomaly_result["facts"] if fact.kind == "event_followthrough"]
    specs: list[EventFollowthroughSpec] = []
    seen: set[tuple[str, str, Direction, int]] = set()
    for fact in facts:
        if fact.lag_days is None:
            continue
        direction: Direction = "strength" if _event_spread_sign(fact) >= 0 else "weakness"
        key = (fact.leader, fact.follower, direction, int(fact.lag_days))
        if key in seen:
            continue
        seen.add(key)
        specs.append(_spec_from_fact(fact, direction))
        if len(specs) >= max_candidates:
            break
    return specs


def run_event_followthrough_lab(
    assets: list[str],
    start: str,
    end: str,
    *,
    max_candidates: int = 6,
    lockbox_pct: float = 0.25,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if not 0.0 < lockbox_pct < 1.0:
        raise ValueError("lockbox_pct must be in (0, 1)")
    clean_assets = _clean_assets(assets)
    if len(clean_assets) < 2:
        raise ValueError("event-followthrough lab needs at least 2 assets")

    full_panel = panel if panel is not None else load_portfolio_panel(clean_assets, start, end)
    train_end, lockbox_start = split_date_range(start, end, lockbox_pct)
    slate = generate_event_followthrough_slate(
        clean_assets,
        start,
        end,
        max_candidates=max_candidates,
        top_anomalies=max(max_candidates * 5, 20),
    )
    train_mask = (full_panel.index >= pd.Timestamp(start)) & (full_panel.index <= pd.Timestamp(train_end))
    lockbox_mask = (full_panel.index >= pd.Timestamp(lockbox_start)) & (full_panel.index <= pd.Timestamp(end))

    results: list[dict[str, Any]] = []
    for spec in slate:
        train = evaluate_event_followthrough_spec(spec, full_panel, train_mask)
        lockbox = evaluate_event_followthrough_spec(spec, full_panel, lockbox_mask)
        stress = stress_event_followthrough_spec(spec, full_panel, lockbox_mask)
        results.append(
            {
                "spec": spec,
                "train": train,
                "lockbox": lockbox,
                "stress": stress,
                "train_pass": _passes_train(train),
                "lockbox_pass": _passes_lockbox(lockbox),
                "robust_pass": stress["verdict"] == "ROBUST",
            }
        )

    train_survivors = [entry for entry in results if entry["train_pass"]]
    lockbox_survivors = [entry for entry in train_survivors if entry["lockbox_pass"]]
    robust_survivors = [entry for entry in lockbox_survivors if entry["robust_pass"]]
    winner = _select_winner(robust_survivors)
    return {
        "assets": clean_assets,
        "start": start,
        "end": end,
        "train_end": train_end,
        "lockbox_start": lockbox_start,
        "lockbox_pct": lockbox_pct,
        "slate": slate,
        "results": results,
        "summary": {
            "pre_registered_candidates": len(slate),
            "train_survivors": len(train_survivors),
            "lockbox_survivors": len(lockbox_survivors),
            "robust_survivors": len(robust_survivors),
            "verdict": _summary_verdict(len(slate), len(train_survivors), len(lockbox_survivors), len(robust_survivors)),
        },
        "winner": winner,
    }


def evaluate_event_followthrough_spec(
    spec: EventFollowthroughSpec,
    panel: pd.DataFrame,
    mask: pd.Series | np.ndarray,
) -> EventFollowthroughMetrics:
    position = event_followthrough_position(spec, panel)
    daily_returns = panel[spec.target].pct_change().fillna(0.0)
    strategy_daily = position.shift(1).fillna(0.0) * daily_returns
    strategy_daily = strategy_daily - position.diff().abs().fillna(0.0) * spec.cost_per_turnover
    segment_strategy = strategy_daily.loc[mask]
    segment_bh = daily_returns.loc[mask]
    segment_position = position.loc[mask]
    return _metrics(segment_strategy, segment_bh, segment_position)


def event_followthrough_position(spec: EventFollowthroughSpec, panel: pd.DataFrame) -> pd.Series:
    if spec.leader not in panel.columns:
        raise ValueError(f"leader {spec.leader} is not in panel")
    if spec.target not in panel.columns:
        raise ValueError(f"target {spec.target} is not in panel")
    zscore = _rolling_event_zscore(
        panel[spec.leader],
        event_window=spec.event_window,
        z_lookback_days=spec.z_lookback_days,
    )
    if spec.direction == "strength":
        event = zscore >= spec.z_threshold
    else:
        event = zscore <= -spec.z_threshold
    return _position_from_events(event, spec.lag_days, spec.hold_days)


def stress_event_followthrough_spec(
    spec: EventFollowthroughSpec,
    panel: pd.DataFrame,
    mask: pd.Series | np.ndarray,
    *,
    lag_offsets: Iterable[int] = (-20, -10, 0, 10, 20),
    hold_values: Iterable[int] = (10, 20, 40),
    threshold_values: Iterable[float] = (0.75, 1.0, 1.25),
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for offset in lag_offsets:
        lag = spec.lag_days + int(offset)
        if lag < 1:
            continue
        for hold in hold_values:
            if hold < 1:
                continue
            for threshold in threshold_values:
                variant = replace(
                    spec,
                    name=f"{spec.name} lag={lag} hold={hold} z={threshold:g}",
                    lag_days=lag,
                    hold_days=int(hold),
                    z_threshold=float(threshold),
                )
                metrics = evaluate_event_followthrough_spec(variant, panel, mask)
                passed = _passes_lockbox(metrics)
                entries.append({"spec": variant, "metrics": metrics, "passed": passed})
    passed_count = sum(1 for entry in entries if entry["passed"])
    pass_rate = passed_count / len(entries) if entries else 0.0
    verdict = "ROBUST" if len(entries) >= 6 and pass_rate >= 0.50 else "FRAGILE"
    return {
        "tested": len(entries),
        "passed": passed_count,
        "pass_rate": pass_rate,
        "verdict": verdict,
        "entries": entries,
    }


def learning_records_from_event_followthrough(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a lab result into append-only structured learnings.

    Each candidate gets one record: winners, losers, and the concrete lesson from
    its failure/success gate. These records intentionally preserve denominators so
    later history summaries can distinguish "one robust survivor out of six" from
    a standalone success story.
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = result["summary"]
    records: list[dict[str, Any]] = []
    for entry in result.get("results", []):
        spec: EventFollowthroughSpec = entry["spec"]
        train: EventFollowthroughMetrics = entry["train"]
        lockbox: EventFollowthroughMetrics = entry["lockbox"]
        stress = entry["stress"]
        status = _learning_status(entry)
        lesson = _learning_lesson(status, spec, lockbox, stress)
        records.append(
            {
                "timestamp": timestamp,
                "mode": "event_followthrough",
                "is_lockbox": False,
                "asset": spec.target,
                "strategy_family": "event_followthrough",
                "verdict": status,
                "learning_status": status,
                "lesson": lesson,
                "mechanism": spec.hypothesis,
                "start_date": result["start"],
                "end_date": result["end"],
                "train_end": result["train_end"],
                "lockbox_start": result["lockbox_start"],
                "params": {
                    "assets": list(result["assets"]),
                    "leader": spec.leader,
                    "target": spec.target,
                    "direction": spec.direction,
                    "z_threshold": spec.z_threshold,
                    "event_window": spec.event_window,
                    "z_lookback_days": spec.z_lookback_days,
                    "lag_days": spec.lag_days,
                    "hold_days": spec.hold_days,
                    "cost_per_turnover": spec.cost_per_turnover,
                },
                "metrics": {
                    "train_annualized_return_pct": train.annualized_return_pct,
                    "train_sharpe": train.sharpe,
                    "train_max_drawdown_pct": train.max_drawdown_pct,
                    "lockbox_annualized_return_pct": lockbox.annualized_return_pct,
                    "lockbox_sharpe": lockbox.sharpe,
                    "lockbox_max_drawdown_pct": lockbox.max_drawdown_pct,
                    "lockbox_buy_hold_annualized_pct": lockbox.buy_hold_annualized_pct,
                    "lockbox_buy_hold_sharpe": lockbox.buy_hold_sharpe,
                    "exposure_pct": lockbox.exposure_pct,
                    "num_entries": lockbox.num_entries,
                },
                "stress_summary": {
                    "tested": stress["tested"],
                    "passed": stress["passed"],
                    "pass_rate": stress["pass_rate"],
                    "verdict": stress["verdict"],
                },
                "gate_denominators": dict(summary),
            }
        )
    return records


def format_event_followthrough_lab(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"Assets: {', '.join(result['assets'])}",
        f"Range: {result['start']} to {result['end']}",
        f"Train end: {result['train_end']}",
        f"Lockbox start: {result['lockbox_start']} ({result['lockbox_pct']:.0%})",
        "",
        f"Pre-registered event strategies: {summary['pre_registered_candidates']}",
        f"Train survivors:                 {summary['train_survivors']}/{summary['pre_registered_candidates']}",
        f"Lockbox survivors:               {summary['lockbox_survivors']}/{summary['train_survivors']}",
        f"Robust stress survivors:         {summary['robust_survivors']}/{summary['lockbox_survivors']}",
        f"VERDICT: {summary['verdict']}",
        "",
        "Per-strategy results:",
    ]
    for idx, entry in enumerate(result["results"], start=1):
        spec: EventFollowthroughSpec = entry["spec"]
        train: EventFollowthroughMetrics = entry["train"]
        lockbox: EventFollowthroughMetrics = entry["lockbox"]
        stress = entry["stress"]
        lines.extend(
            [
                f"{idx}. {spec.name}",
                f"   Rule: {spec.direction} event in {spec.leader}; wait {spec.lag_days}d; hold {spec.target} for {spec.hold_days}d; z≥{spec.z_threshold:g}; cost={spec.cost_per_turnover:.2%}/turnover",
                f"   Train:   ann {train.annualized_return_pct:+.1f}%, Sharpe {_fmt_optional(train.sharpe)}, maxDD {train.max_drawdown_pct:+.1f}%, exposure {train.exposure_pct:.1f}%, entries {train.num_entries}",
                f"   Lockbox: ann {lockbox.annualized_return_pct:+.1f}%, Sharpe {_fmt_optional(lockbox.sharpe)}, maxDD {lockbox.max_drawdown_pct:+.1f}%, exposure {lockbox.exposure_pct:.1f}%, entries {lockbox.num_entries}; buy-hold ann {lockbox.buy_hold_annualized_pct:+.1f}%, Sharpe {_fmt_optional(lockbox.buy_hold_sharpe)}",
                f"   Stress:  {stress['passed']}/{stress['tested']} variants pass ({stress['pass_rate']:.0%}) → {stress['verdict']}",
            ]
        )
    if result.get("winner") is not None:
        winner = result["winner"]
        spec = winner["spec"]
        lines.extend(["", f"Winner candidate: {spec.name}"])
    lines.append("")
    lines.append(
        "Interpretation: generated from anomaly facts, but only candidates surviving train, lockbox, and neighborhood stress should be paper-traded."
    )
    return "\n".join(lines)


def _spec_from_fact(fact: AnomalyFact, direction: Direction) -> EventFollowthroughSpec:
    direction_label = "strength" if direction == "strength" else "weakness"
    name = f"{fact.leader} {direction_label} -> {fact.follower} lag{fact.lag_days}"
    return EventFollowthroughSpec(
        name=name,
        leader=fact.leader,
        target=fact.follower,
        direction=direction,
        z_threshold=1.0,
        lag_days=int(fact.lag_days or 1),
        hold_days=20,
        hypothesis=(
            f"A {direction_label} shock in {fact.leader} transmits with delay into "
            f"{fact.follower}; if structural, nearby lags/holds should also work."
        ),
    )


def _event_spread_sign(fact: AnomalyFact) -> float:
    # Event-followthrough facts store positive-event minus negative-event spread in
    # the fact/control text and in train/holdout scores. Prefer the stable scores.
    if fact.train_score is not None and fact.holdout_score is not None:
        return fact.train_score + fact.holdout_score
    return fact.score


def _rolling_event_zscore(
    series: pd.Series,
    *,
    event_window: int,
    z_lookback_days: int,
) -> pd.Series:
    event_return = series.astype(float).pct_change(event_window)
    mean = event_return.rolling(z_lookback_days, min_periods=80).mean()
    std = event_return.rolling(z_lookback_days, min_periods=80).std().replace(0.0, np.nan)
    return (event_return - mean) / std


def _position_from_events(event: pd.Series, lag_days: int, hold_days: int) -> pd.Series:
    starts = event.astype("boolean").shift(lag_days).fillna(False).astype(bool)
    active = pd.Series(False, index=event.index)
    values = starts.to_numpy()
    for idx, is_start in enumerate(values):
        if is_start:
            active.iloc[idx : min(idx + hold_days, len(active))] = True
    return active.astype(float)


def _metrics(strategy_daily: pd.Series, buy_hold_daily: pd.Series, position: pd.Series) -> EventFollowthroughMetrics:
    clean_strategy = strategy_daily.dropna().astype(float)
    clean_bh = buy_hold_daily.reindex(clean_strategy.index).fillna(0.0).astype(float)
    clean_position = position.reindex(clean_strategy.index).fillna(0.0).astype(float)
    strategy_curve = (1.0 + clean_strategy).cumprod()
    bh_curve = (1.0 + clean_bh).cumprod()
    n = len(clean_strategy)
    ann = _annualized_return(strategy_curve, n)
    bh_ann = _annualized_return(bh_curve, n)
    sharpe = _sharpe(clean_strategy)
    bh_sharpe = _sharpe(clean_bh)
    max_dd = _max_drawdown(strategy_curve)
    bh_max_dd = _max_drawdown(bh_curve)
    return EventFollowthroughMetrics(
        total_return_pct=float((strategy_curve.iloc[-1] - 1.0) * 100) if n else 0.0,
        annualized_return_pct=ann * 100,
        sharpe=sharpe,
        max_drawdown_pct=max_dd * 100,
        exposure_pct=float((clean_position > 0).mean() * 100) if n else 0.0,
        num_entries=int(((clean_position > 0) & (clean_position.shift(1).fillna(0) <= 0)).sum()),
        buy_hold_return_pct=float((bh_curve.iloc[-1] - 1.0) * 100) if n else 0.0,
        buy_hold_annualized_pct=bh_ann * 100,
        buy_hold_sharpe=bh_sharpe,
        buy_hold_max_drawdown_pct=bh_max_dd * 100,
        beats_buy_hold_sharpe=(sharpe or -999.0) > (bh_sharpe or -999.0),
    )


def _annualized_return(curve: pd.Series, n_obs: int) -> float:
    if n_obs < 1 or curve.empty or curve.iloc[-1] <= 0:
        return 0.0
    return float(curve.iloc[-1] ** (252 / n_obs) - 1.0)


def _sharpe(daily: pd.Series) -> float | None:
    std = float(daily.std())
    if std == 0.0 or np.isnan(std):
        return None
    return float(daily.mean() / std * np.sqrt(252))


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    return float((curve / curve.cummax() - 1.0).min())


def _learning_status(entry: dict[str, Any]) -> str:
    if entry.get("robust_pass"):
        return "winner"
    if entry.get("lockbox_pass"):
        return "fragile_lockbox_survivor"
    if entry.get("train_pass"):
        return "lockbox_loser"
    return "train_loser"


def _learning_lesson(
    status: str,
    spec: EventFollowthroughSpec,
    lockbox: EventFollowthroughMetrics,
    stress: dict[str, Any],
) -> str:
    rule = f"{spec.leader} {spec.direction} → {spec.target} lag {spec.lag_days} hold {spec.hold_days}"
    if status == "winner":
        return (
            f"{rule} survived train, lockbox, and neighborhood stress; mechanism is worth "
            "forward paper trading, not further historical tuning."
        )
    if status == "fragile_lockbox_survivor":
        return (
            f"{rule} passed lockbox but failed neighborhood stress "
            f"({stress['passed']}/{stress['tested']} variants); treat as parameter-fragile."
        )
    if status == "lockbox_loser":
        return (
            f"{rule} looked acceptable on train but did not beat the lockbox gate; "
            f"lockbox Sharpe {_fmt_optional(lockbox.sharpe)} vs buy-hold {_fmt_optional(lockbox.buy_hold_sharpe)}."
        )
    return f"{rule} failed the train gate; reject before spending lockbox/stress attention."


def _passes_train(metrics: EventFollowthroughMetrics) -> bool:
    return (
        metrics.total_return_pct > 0
        and (metrics.sharpe or -999.0) >= 0.50
        and metrics.max_drawdown_pct > -75.0
        and metrics.num_entries >= 5
    )


def _passes_lockbox(metrics: EventFollowthroughMetrics) -> bool:
    return (
        metrics.total_return_pct > 0
        and (metrics.sharpe or -999.0) >= 0.80
        and metrics.beats_buy_hold_sharpe
        and metrics.max_drawdown_pct > -50.0
        and metrics.num_entries >= 2
    )


def _select_winner(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda entry: (
            entry["stress"]["pass_rate"],
            entry["lockbox"].sharpe or -999.0,
            entry["lockbox"].annualized_return_pct,
        ),
    )


def _summary_verdict(slate: int, train: int, lockbox: int, robust: int) -> str:
    if slate == 0:
        return "REJECTED_NO_EVENT_CANDIDATES"
    if train == 0:
        return "REJECTED_NO_TRAIN_SURVIVORS"
    if lockbox == 0:
        return "REJECTED_NO_LOCKBOX_SURVIVORS"
    if robust == 0:
        return "REJECTED_FRAGILE_LOCKBOX_SURVIVORS"
    return "PAPER_TRADE_CANDIDATE"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _clean_assets(assets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in assets:
        asset = raw.strip()
        if asset and asset.lower() not in seen:
            seen.add(asset.lower())
            out.append(asset)
    return out
