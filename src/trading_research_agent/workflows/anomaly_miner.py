"""Deterministic market anomaly mining.

This module produces concrete empirical facts for the hypothesis generator. It is
not a strategy optimizer and does not select parameters for trading. It scans a
fixed asset/date universe for specific non-generic anomalies such as lead/lag
relationships, event-conditioned follow-through, and regime-conditioned
asymmetries, then reports denominators and controls for skeptical follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erfc, log, sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd

from trading_research_agent.tools.data_loader import load_portfolio_panel


@dataclass(frozen=True)
class AnomalyFact:
    kind: str
    leader: str
    follower: str
    lag_days: int | None
    score: float
    fact: str
    control: str
    # Evidence fields are intentionally lightweight stdlib math, not scipy-only.
    trials: int = 1
    p_value: float | None = None
    adjusted_p_value: float | None = None
    train_score: float | None = None
    holdout_score: float | None = None


def mine_anomalies(
    assets: list[str],
    start: str,
    end: str,
    *,
    top_n: int = 12,
    lags: tuple[int, ...] = (5, 10, 20, 40, 60),
    horizon: int = 20,
) -> dict[str, Any]:
    clean_assets = _clean_assets(assets)
    if len(clean_assets) < 2:
        raise ValueError("mine_anomalies requires at least 2 assets")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    panel = load_portfolio_panel(clean_assets, start, end)
    lead_lag = mine_lead_lag_anomalies(panel, lags=lags, horizon=horizon, top_n=top_n)
    events = mine_event_followthrough_anomalies(panel, lags=lags, horizon=horizon, top_n=top_n)
    regimes = mine_regime_anomalies(panel, top_n=top_n)
    facts = [*lead_lag, *events, *regimes]
    facts = sorted(facts, key=_rank_fact, reverse=True)[:top_n]
    return {
        "assets": clean_assets,
        "start": start,
        "end": end,
        "rows": len(panel),
        "tests_scanned": _tests_scanned(clean_assets, lags),
        "facts": facts,
    }


def mine_lead_lag_anomalies(
    panel: pd.DataFrame,
    *,
    lags: Iterable[int] = (5, 10, 20, 40, 60),
    horizon: int = 20,
    min_abs_corr: float = 0.18,
    max_adjusted_p: float = 0.25,
    top_n: int = 12,
) -> list[AnomalyFact]:
    """Find pairs where leader horizon returns correlate with lagged follower returns.

    At lag L, we correlate leader t..t+h return with follower t+L..t+L+h return.
    A first-half/second-half control is included so the hypothesis generator sees
    whether the relationship is stable or regime-specific. Correlation p-values
    are approximate Fisher-z p-values and Bonferroni-adjusted by the scan count;
    they are a false-positive throttle, not proof of edge.
    """
    lag_list = tuple(int(lag) for lag in lags)
    returns = _forward_returns(panel, horizon)
    trials = _pair_lag_trials(panel.columns, lag_list)
    facts: list[AnomalyFact] = []
    for leader in returns.columns:
        for follower in returns.columns:
            if leader == follower:
                continue
            for lag in lag_list:
                pair = pd.concat(
                    [returns[leader], returns[follower].shift(-lag)], axis=1
                ).dropna()
                if len(pair) < 80:
                    continue
                corr = _safe_corr(pair.iloc[:, 0], pair.iloc[:, 1])
                if corr is None or abs(corr) < min_abs_corr:
                    continue
                p_value = _corr_p_value(corr, len(pair))
                adjusted_p = min(1.0, p_value * trials)
                if adjusted_p > max_adjusted_p:
                    continue
                first, second = _split_corr(pair)
                stability = _same_sign_min_abs(corr, first, second)
                facts.append(
                    AnomalyFact(
                        kind="lead_lag",
                        leader=leader,
                        follower=follower,
                        lag_days=lag,
                        score=float(abs(corr) * (0.5 + stability)),
                        fact=(
                            f"{leader} {horizon}d returns lead {follower} {horizon}d returns "
                            f"at lag {lag} with corr {corr:+.2f}."
                        ),
                        control=(
                            f"First-half corr {first:+.2f}; second-half corr {second:+.2f}; "
                            f"n={len(pair)} aligned observations; p≈{p_value:.3g}; "
                            f"Bonferroni p≈{adjusted_p:.3g} over {trials} lead/lag tests."
                        ),
                        trials=trials,
                        p_value=p_value,
                        adjusted_p_value=adjusted_p,
                        train_score=first,
                        holdout_score=second,
                    )
                )
    return sorted(facts, key=_rank_fact, reverse=True)[:top_n]


def mine_event_followthrough_anomalies(
    panel: pd.DataFrame,
    *,
    lags: Iterable[int] = (5, 10, 20, 40, 60),
    horizon: int = 20,
    event_window: int = 20,
    z_threshold: float = 1.0,
    min_events: int = 18,
    min_spread_pct: float = 1.0,
    top_n: int = 12,
) -> list[AnomalyFact]:
    """Find conditional follow-through after unusually strong/weak leader moves.

    This is closer to a tradeable hypothesis than raw correlation: if a leader has
    an unusually positive/negative `event_window` return, what happens to the
    follower over `horizon` days after `lag` days? The control compares positive
    vs negative leader events, reports event counts, and checks train/holdout sign
    consistency. It still only emits facts; validation remains lockbox/stress.
    """
    lag_list = tuple(int(lag) for lag in lags)
    returns = _forward_returns(panel, horizon)
    leader_event_return = panel.pct_change(event_window)
    rolling_mean = leader_event_return.rolling(252, min_periods=80).mean()
    rolling_std = leader_event_return.rolling(252, min_periods=80).std().replace(0.0, np.nan)
    zscores = (leader_event_return - rolling_mean) / rolling_std
    trials = _pair_lag_trials(panel.columns, lag_list)
    facts: list[AnomalyFact] = []

    for leader in panel.columns:
        positive_event = zscores[leader] >= z_threshold
        negative_event = zscores[leader] <= -z_threshold
        for follower in panel.columns:
            if leader == follower:
                continue
            for lag in lag_list:
                future = returns[follower].shift(-lag)
                pos = future.loc[positive_event].dropna()
                neg = future.loc[negative_event].dropna()
                if len(pos) < min_events or len(neg) < min_events:
                    continue
                spread_pct = float((pos.mean() - neg.mean()) * 100)
                if abs(spread_pct) < min_spread_pct:
                    continue
                sample = pd.concat(
                    [
                        pd.Series(1.0, index=pos.index),
                        pd.Series(-1.0, index=neg.index),
                    ]
                ).sort_index()
                outcome = pd.concat([pos, neg]).sort_index().reindex(sample.index)
                train_spread, holdout_spread = _split_event_spread(sample, outcome)
                if not _same_direction(spread_pct, train_spread, holdout_spread):
                    continue
                p_value = _welch_p_value(pos, neg)
                adjusted_p = min(1.0, p_value * trials)
                facts.append(
                    AnomalyFact(
                        kind="event_followthrough",
                        leader=leader,
                        follower=follower,
                        lag_days=lag,
                        score=float(abs(spread_pct) * (0.5 + _sign_stability(spread_pct, train_spread, holdout_spread))),
                        fact=(
                            f"After {leader} {event_window}d z-score ≥ {z_threshold:.1f}, "
                            f"{follower} next {horizon}d return after lag {lag} averaged {pos.mean()*100:+.2f}%; "
                            f"after z-score ≤ -{z_threshold:.1f}, it averaged {neg.mean()*100:+.2f}%."
                        ),
                        control=(
                            f"Positive events n={len(pos)}, negative events n={len(neg)}; "
                            f"spread {spread_pct:+.2f} pct points; train spread {train_spread:+.2f}; "
                            f"holdout spread {holdout_spread:+.2f}; Welch p≈{p_value:.3g}; "
                            f"Bonferroni p≈{adjusted_p:.3g} over {trials} event tests."
                        ),
                        trials=trials,
                        p_value=p_value,
                        adjusted_p_value=adjusted_p,
                        train_score=train_spread,
                        holdout_score=holdout_spread,
                    )
                )
    return sorted(facts, key=_rank_fact, reverse=True)[:top_n]


def mine_regime_anomalies(
    panel: pd.DataFrame,
    *,
    trend_window: int = 63,
    forward_horizon: int = 20,
    min_spread_pct: float = 2.0,
    max_adjusted_p: float = 0.35,
    top_n: int = 12,
) -> list[AnomalyFact]:
    """Find target assets with different forward returns after signal up/down regimes."""
    future = _forward_returns(panel, forward_horizon)
    trend = panel.pct_change(trend_window)
    trials = len(panel.columns) * (len(panel.columns) - 1)
    facts: list[AnomalyFact] = []
    for signal in panel.columns:
        up = trend[signal] > 0
        down = trend[signal] <= 0
        for target in panel.columns:
            if signal == target:
                continue
            up_returns = future.loc[up, target].dropna()
            down_returns = future.loc[down, target].dropna()
            if len(up_returns) < 40 or len(down_returns) < 40:
                continue
            up_mean = float(up_returns.mean() * 100)
            down_mean = float(down_returns.mean() * 100)
            spread = up_mean - down_mean
            if abs(spread) < min_spread_pct:
                continue
            p_value = _welch_p_value(up_returns, down_returns)
            adjusted_p = min(1.0, p_value * trials)
            if adjusted_p > max_adjusted_p:
                continue
            sample = pd.concat(
                [
                    pd.Series(1.0, index=up_returns.index),
                    pd.Series(-1.0, index=down_returns.index),
                ]
            ).sort_index()
            outcome = pd.concat([up_returns, down_returns]).sort_index().reindex(sample.index)
            train_spread, holdout_spread = _split_event_spread(sample, outcome)
            facts.append(
                AnomalyFact(
                    kind="regime_split",
                    leader=signal,
                    follower=target,
                    lag_days=None,
                    score=float(abs(spread) * (0.5 + _sign_stability(spread, train_spread, holdout_spread))),
                    fact=(
                        f"{target} forward {forward_horizon}d return differs after {signal} regimes: "
                        f"{up_mean:+.2f}% after {trend_window}d up-trend vs {down_mean:+.2f}% after down-trend."
                    ),
                    control=(
                        f"Spread {spread:+.2f} pct points; up n={len(up_returns)}, down n={len(down_returns)}; "
                        f"train spread {train_spread:+.2f}; holdout spread {holdout_spread:+.2f}; "
                        f"Welch p≈{p_value:.3g}; Bonferroni p≈{adjusted_p:.3g} over {trials} regime tests."
                    ),
                    trials=trials,
                    p_value=p_value,
                    adjusted_p_value=adjusted_p,
                    train_score=train_spread,
                    holdout_score=holdout_spread,
                )
            )
    return sorted(facts, key=_rank_fact, reverse=True)[:top_n]


def format_anomaly_report(result: dict[str, Any]) -> str:
    facts: list[AnomalyFact] = result.get("facts", [])
    lines = [
        f"Assets: {', '.join(result.get('assets', []))}",
        f"Range: {result.get('start')} to {result.get('end')}",
    ]
    if "rows" in result:
        lines.append(f"Aligned rows: {result['rows']}")
    if "tests_scanned" in result:
        lines.append(f"Approx tests scanned: {result['tests_scanned']}")
    lines.extend(
        [
            f"Facts reported: {len(facts)}",
            "",
            "These are anomaly candidates, not trading edges. Edge requires: structural story → frozen PortfolioSpec → train screen → lockbox → robustness stress → forward paper trading.",
            "",
        ]
    )
    for idx, fact in enumerate(facts, start=1):
        lag = f", lag={fact.lag_days}d" if fact.lag_days is not None else ""
        stats = _format_fact_stats(fact)
        lines.extend(
            [
                f"{idx}. {fact.kind}{lag}: {fact.leader} → {fact.follower} | score={fact.score:.2f}{stats}",
                f"   Fact: {fact.fact}",
                f"   Control: {fact.control}",
            ]
        )
    return "\n".join(lines).rstrip()


def _forward_returns(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    prices = panel.astype(float).sort_index()
    return prices.shift(-horizon) / prices - 1.0


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 20:
        return None
    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if corr is None or np.isnan(corr) or np.isinf(corr):
        return None
    return float(corr)


def _split_corr(pair: pd.DataFrame) -> tuple[float, float]:
    split = len(pair) // 2
    first = _safe_corr(pair.iloc[:split, 0], pair.iloc[:split, 1])
    second = _safe_corr(pair.iloc[split:, 0], pair.iloc[split:, 1])
    return (first or 0.0, second or 0.0)


def _corr_p_value(corr: float, n: int) -> float:
    if n <= 3:
        return 1.0
    clipped = max(-0.999999, min(0.999999, corr))
    fisher_z = 0.5 * log((1.0 + clipped) / (1.0 - clipped)) * sqrt(n - 3)
    return _two_sided_normal_p(fisher_z)


def _welch_p_value(a: pd.Series, b: pd.Series) -> float:
    a = pd.Series(a, dtype=float).dropna()
    b = pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2:
        return 1.0
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    denom = sqrt(var_a / len(a) + var_b / len(b))
    if denom == 0.0 or np.isnan(denom):
        return 1.0
    t_stat = float((a.mean() - b.mean()) / denom)
    # Normal tail approximation is conservative enough for scan throttling here.
    return _two_sided_normal_p(t_stat)


def _two_sided_normal_p(z: float) -> float:
    return float(erfc(abs(z) / sqrt(2.0)))


def _split_event_spread(labels: pd.Series, outcome: pd.Series) -> tuple[float, float]:
    data = pd.concat([labels.rename("label"), outcome.rename("outcome")], axis=1).dropna()
    split = len(data) // 2
    return (_event_spread(data.iloc[:split]), _event_spread(data.iloc[split:]))


def _event_spread(data: pd.DataFrame) -> float:
    pos = data.loc[data["label"] > 0, "outcome"]
    neg = data.loc[data["label"] < 0, "outcome"]
    if pos.empty or neg.empty:
        return 0.0
    return float((pos.mean() - neg.mean()) * 100)


def _same_sign_min_abs(full: float, first: float, second: float) -> float:
    if _same_direction(full, first, second):
        return min(abs(first), abs(second), abs(full))
    return 0.0


def _same_direction(full: float, first: float, second: float) -> bool:
    if full == 0.0 or first == 0.0 or second == 0.0:
        return False
    sign = np.sign(full)
    return bool(np.sign(first) == sign and np.sign(second) == sign)


def _sign_stability(full: float, first: float, second: float) -> float:
    if not _same_direction(full, first, second):
        return 0.0
    denom = max(abs(full), 1e-12)
    return min(1.0, min(abs(first), abs(second)) / denom)


def _rank_fact(fact: AnomalyFact) -> float:
    p_bonus = 1.0
    if fact.adjusted_p_value is not None:
        p_bonus += max(0.0, -log(max(fact.adjusted_p_value, 1e-12))) / 10.0
    stability_bonus = 0.0
    if fact.train_score is not None and fact.holdout_score is not None:
        stability_bonus = _sign_stability(fact.score, fact.train_score, fact.holdout_score)
    return abs(fact.score) * (1.0 + stability_bonus) * p_bonus


def _pair_lag_trials(columns: Iterable[str], lags: Iterable[int]) -> int:
    cols = list(columns)
    return max(1, len(cols) * (len(cols) - 1) * len(tuple(lags)))


def _tests_scanned(assets: list[str], lags: Iterable[int]) -> int:
    pair_lags = _pair_lag_trials(assets, lags)
    regimes = len(assets) * (len(assets) - 1)
    # lead/lag + event follow-through + regime split
    return pair_lags * 2 + regimes


def _format_fact_stats(fact: AnomalyFact) -> str:
    parts: list[str] = []
    if fact.adjusted_p_value is not None:
        parts.append(f"adj_p≈{fact.adjusted_p_value:.3g}")
    if fact.train_score is not None and fact.holdout_score is not None:
        parts.append(f"train={fact.train_score:+.2f}")
        parts.append(f"holdout={fact.holdout_score:+.2f}")
    return " | " + ", ".join(parts) if parts else ""


def _clean_assets(assets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in assets:
        asset = raw.strip()
        if asset and asset.lower() not in seen:
            seen.add(asset.lower())
            out.append(asset)
    return out
