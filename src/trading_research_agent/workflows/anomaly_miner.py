"""Deterministic market anomaly mining.

This module produces concrete empirical facts for the hypothesis generator. It is
not a strategy optimizer and does not select parameters for trading. It scans a
fixed asset/date universe for specific non-generic anomalies such as lead/lag
relationships and regime-conditioned asymmetries, then reports denominators and
controls for skeptical follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    facts = [
        *mine_lead_lag_anomalies(panel, lags=lags, horizon=horizon, top_n=top_n),
        *mine_regime_anomalies(panel, top_n=top_n),
    ]
    facts = sorted(facts, key=lambda f: abs(f.score), reverse=True)[:top_n]
    return {
        "assets": clean_assets,
        "start": start,
        "end": end,
        "rows": len(panel),
        "facts": facts,
    }


def mine_lead_lag_anomalies(
    panel: pd.DataFrame,
    *,
    lags: Iterable[int] = (5, 10, 20, 40, 60),
    horizon: int = 20,
    min_abs_corr: float = 0.18,
    top_n: int = 12,
) -> list[AnomalyFact]:
    """Find pairs where leader horizon returns correlate with lagged follower returns.

    At lag L, we correlate leader t..t+h return with follower t+L..t+L+h return.
    A first-half/second-half control is included so the hypothesis generator sees
    whether the relationship is stable or regime-specific.
    """
    returns = _forward_returns(panel, horizon)
    facts: list[AnomalyFact] = []
    for leader in returns.columns:
        for follower in returns.columns:
            if leader == follower:
                continue
            for lag in lags:
                pair = pd.concat(
                    [returns[leader], returns[follower].shift(-int(lag))], axis=1
                ).dropna()
                if len(pair) < 80:
                    continue
                corr = _safe_corr(pair.iloc[:, 0], pair.iloc[:, 1])
                if corr is None or abs(corr) < min_abs_corr:
                    continue
                first, second = _split_corr(pair)
                facts.append(
                    AnomalyFact(
                        kind="lead_lag",
                        leader=leader,
                        follower=follower,
                        lag_days=int(lag),
                        score=float(abs(corr)),
                        fact=(
                            f"{leader} {horizon}d returns lead {follower} {horizon}d returns "
                            f"at lag {lag} with corr {corr:+.2f}."
                        ),
                        control=(
                            f"First-half corr {first:+.2f}; second-half corr {second:+.2f}; "
                            f"n={len(pair)} aligned observations."
                        ),
                    )
                )
    return sorted(facts, key=lambda f: abs(f.score), reverse=True)[:top_n]


def mine_regime_anomalies(
    panel: pd.DataFrame,
    *,
    trend_window: int = 63,
    forward_horizon: int = 20,
    min_spread_pct: float = 2.0,
    top_n: int = 12,
) -> list[AnomalyFact]:
    """Find target assets with different forward returns after signal up/down regimes."""
    future = _forward_returns(panel, forward_horizon)
    trend = panel.pct_change(trend_window)
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
            facts.append(
                AnomalyFact(
                    kind="regime_split",
                    leader=signal,
                    follower=target,
                    lag_days=None,
                    score=float(abs(spread)),
                    fact=(
                        f"{target} forward {forward_horizon}d return differs after {signal} regimes: "
                        f"{up_mean:+.2f}% after {trend_window}d up-trend vs {down_mean:+.2f}% after down-trend."
                    ),
                    control=(
                        f"Spread {spread:+.2f} pct points; up n={len(up_returns)}, down n={len(down_returns)}."
                    ),
                )
            )
    return sorted(facts, key=lambda f: abs(f.score), reverse=True)[:top_n]


def format_anomaly_report(result: dict[str, Any]) -> str:
    facts: list[AnomalyFact] = result.get("facts", [])
    lines = [
        f"Assets: {', '.join(result.get('assets', []))}",
        f"Range: {result.get('start')} to {result.get('end')}",
    ]
    if "rows" in result:
        lines.append(f"Aligned rows: {result['rows']}")
    lines.extend(
        [
            f"Facts reported: {len(facts)}",
            "",
            "These are anomaly candidates, not trading edges. Use them as hard context for hypothesis generation; then try to falsify them with lockbox/stress tests.",
            "",
        ]
    )
    for idx, fact in enumerate(facts, start=1):
        lag = f", lag={fact.lag_days}d" if fact.lag_days is not None else ""
        lines.extend(
            [
                f"{idx}. {fact.kind}{lag}: {fact.leader} → {fact.follower} | score={fact.score:.2f}",
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


def _clean_assets(assets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in assets:
        asset = raw.strip()
        if asset and asset.lower() not in seen:
            seen.add(asset.lower())
            out.append(asset)
    return out
