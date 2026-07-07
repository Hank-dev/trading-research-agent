#!/usr/bin/env python3
"""Canonical rerun for the narrowed EMA/Donchian/ATR trend-pullback slate."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trading_research_agent.strategies.trend_pullback import (  # noqa: E402
    TrendPullbackParams,
    simulate_trades,
    split_periods,
    summarize_trades,
)


OUTPUT_DIR = Path("/home/johannes/trading-research-agent/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "canonical_trend_pullback_20260617.csv"
YEARLY_CSV = OUTPUT_DIR / "canonical_trend_pullback_yearly_20260617.csv"
REPORT_MD = OUTPUT_DIR / "canonical_trend_pullback_20260617.md"

START = "2010-01-01"
END = "2026-06-18"
LOCKBOX_START = "2022-01-01"
MIN_LOCKBOX_TRADES = 8
COST_BPS_GRID = [0.0, 2.0, 5.0, 10.0, 20.0]


@dataclass(frozen=True)
class Candidate:
    name: str
    ticker: str
    instrument_note: str


CANDIDATES = [
    Candidate("WS30", "^DJI", "Dow Jones cash index proxy, not directly tradable"),
    Candidate("J225", "^N225", "Nikkei 225 cash index proxy, not directly tradable"),
    Candidate("SPA35", "^IBEX", "IBEX 35 cash index proxy, not directly tradable"),
    Candidate("XAU_FUT", "GC=F", "COMEX gold futures proxy, not spot XAUUSD"),
    Candidate("XAU_SPOT", "XAUUSD=X", "Yahoo spot gold FX proxy if available"),
]


def load_daily(ticker: str) -> pd.DataFrame:
    """Load adjusted daily OHLC from Yahoo Finance.

    auto_adjust=True keeps OHLC internally consistent for daily bars. Futures
    are still futures proxies, and cash index proxies are still not executable
    instruments.
    """
    df = yf.download(
        ticker,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close"]].dropna().sort_index()


def positive_year_fraction(df: pd.DataFrame, params: TrendPullbackParams) -> tuple[float, int, int]:
    rows = []
    for year, part in df.groupby(df.index.year):
        if year < 2012:
            continue
        trades = simulate_trades(part, params)
        summary = summarize_trades(trades)
        if summary["trades"] == 0:
            continue
        rows.append((year, summary))
    if not rows:
        return np.nan, 0, 0
    positives = sum(1 for _, s in rows if s["total_r"] > 0)
    return positives / len(rows) * 100.0, positives, len(rows)


def row_from_summary(prefix: str, summary: dict) -> dict:
    return {
        f"{prefix}_trades": summary["trades"],
        f"{prefix}_net_return_pct": summary["net_return_pct"],
        f"{prefix}_total_r": summary["total_r"],
        f"{prefix}_win_rate": summary["win_rate"],
        f"{prefix}_profit_factor": summary["profit_factor"],
        f"{prefix}_max_dd_pct": summary["max_dd_pct"],
        f"{prefix}_avg_hold_bars": summary["avg_hold_bars"],
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    yearly_rows: list[dict] = []

    for candidate in CANDIDATES:
        print(f"Downloading {candidate.name} ({candidate.ticker})...")
        data = load_daily(candidate.ticker)
        if data.empty:
            summary_rows.append(
                {
                    "name": candidate.name,
                    "ticker": candidate.ticker,
                    "status": "no_data",
                    "instrument_note": candidate.instrument_note,
                }
            )
            continue

        is_data = split_periods(data, START, "2021-12-31")
        lockbox_data = split_periods(data, LOCKBOX_START, END)

        for ema_period in [100, 150]:
            for trail_atr in [3.0, 4.0]:
                for cost_bps in COST_BPS_GRID:
                    params = TrendPullbackParams(
                        side="long",
                        ema_period=ema_period,
                        sl_atr=1.0,
                        trail_atr=trail_atr,
                        cost_bps_per_side=cost_bps,
                    )
                    all_trades = simulate_trades(data, params)
                    is_trades = simulate_trades(is_data, params)
                    lockbox_trades = simulate_trades(lockbox_data, params)

                    all_summary = summarize_trades(all_trades)
                    is_summary = summarize_trades(is_trades)
                    lockbox_summary = summarize_trades(lockbox_trades)
                    pos_year_pct, pos_years, tested_years = positive_year_fraction(data, params)

                    row = {
                        "name": candidate.name,
                        "ticker": candidate.ticker,
                        "instrument_note": candidate.instrument_note,
                        "timeframe": "1d",
                        "side": "long",
                        "ema_period": ema_period,
                        "sl_atr": 1.0,
                        "trail_atr": trail_atr,
                        "cost_bps_per_side": cost_bps,
                        "round_trip_bps": 2.0 * cost_bps,
                        "bars": len(data),
                        "start": data.index.min().date().isoformat(),
                        "end": data.index.max().date().isoformat(),
                        "positive_year_pct": pos_year_pct,
                        "positive_years": pos_years,
                        "tested_years": tested_years,
                        "status": "ok",
                    }
                    row.update(row_from_summary("full", all_summary))
                    row.update(row_from_summary("is", is_summary))
                    row.update(row_from_summary("lockbox", lockbox_summary))
                    row["passes_lockbox_10bps"] = (
                        cost_bps == 10.0
                        and lockbox_summary["trades"] >= MIN_LOCKBOX_TRADES
                        and lockbox_summary["total_r"] > 0
                        and lockbox_summary["profit_factor"] > 1.1
                        and pos_year_pct >= 50.0
                    )
                    summary_rows.append(row)

                    if cost_bps == 10.0:
                        for year, part in data.groupby(data.index.year):
                            if year < 2012:
                                continue
                            year_summary = summarize_trades(simulate_trades(part, params))
                            yearly_rows.append(
                                {
                                    "name": candidate.name,
                                    "ticker": candidate.ticker,
                                    "ema_period": ema_period,
                                    "trail_atr": trail_atr,
                                    "cost_bps_per_side": cost_bps,
                                    "year": year,
                                    **year_summary,
                                }
                            )

    return pd.DataFrame(summary_rows), pd.DataFrame(yearly_rows)


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    focus = ok[ok["cost_bps_per_side"] == 10.0].copy()
    focus = focus.sort_values(
        ["passes_lockbox_10bps", "lockbox_total_r", "positive_year_pct"],
        ascending=[False, False, False],
    )

    passed = focus[focus["passes_lockbox_10bps"]]
    lines = [
        "# Canonical Trend-Pullback Rerun - 2026-06-17",
        "",
        "Single canonical implementation for the narrowed slate only.",
        "",
        "Rules:",
        "- Long only",
        "- Daily bars from Yahoo Finance with auto_adjust=True",
        "- Entry: close > EMA, low touched Donchian-21 lower support 2 bars ago, close recovered above the channel",
        "- Fill: next daily open after the signal bar",
        "- Stop: 1 ATR(20)",
        "- Exit: ATR(20) trailing stop at 3 or 4 ATR",
        "- Costs: 0 / 2 / 5 / 10 / 20 bps per side",
        f"- In-sample: {START} through 2021-12-31",
        f"- Lockbox: {LOCKBOX_START} through {END}",
        "",
        f"Denominator: {len(ok)} parameter/cost rows from the narrowed slate.",
        f"At 10 bps per side: {len(focus)} rows tested, {len(passed)} passed the canonical lockbox gate.",
        "",
        "Important caveat: this is a cleaned confirmation rerun, not a fresh untouched lockbox.",
        "The slate was selected after earlier screening on overlapping history, so these results are",
        "useful for falsifying contaminated candidates, but they are not final validation.",
        "",
        "Pass gate at 10 bps per side:",
        f"- lockbox trades >= {MIN_LOCKBOX_TRADES}",
        "- lockbox total R > 0",
        "- lockbox profit factor > 1.1",
        "- >= 50% positive calendar-year windows",
        "",
        "## 10 bps per-side results",
        "",
    ]

    display_cols = [
        "name",
        "ticker",
        "ema_period",
        "trail_atr",
        "lockbox_trades",
        "lockbox_total_r",
        "lockbox_profit_factor",
        "lockbox_net_return_pct",
        "positive_year_pct",
        "passes_lockbox_10bps",
        "instrument_note",
    ]
    lines.append("```")
    lines.append(focus[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    lines.append("```")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- WS30/J225/SPA35 are cash index proxies, not directly executable instruments.",
            "- XAU_FUT is COMEX gold futures, not spot XAUUSD.",
            "- XAU_SPOT is included only if Yahoo returns usable XAUUSD=X data.",
            "- This fixes the earlier problem where cost, walk-forward, and plateau scripts used slightly different signal logic.",
        ]
    )

    if not yearly.empty:
        yearly_focus = yearly[yearly["cost_bps_per_side"] == 10.0].copy()
        yearly_focus = yearly_focus.sort_values(["name", "ema_period", "trail_atr", "year"])
        lines.extend(
            [
                "",
                "## Yearly 10 bps detail",
                "",
                "```",
                yearly_focus[
                    [
                        "name",
                        "ema_period",
                        "trail_atr",
                        "year",
                        "trades",
                        "total_r",
                        "profit_factor",
                        "net_return_pct",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.2f}"),
                "```",
            ]
        )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary, yearly = evaluate()
    summary.to_csv(SUMMARY_CSV, index=False)
    yearly.to_csv(YEARLY_CSV, index=False)
    write_report(summary, yearly)

    focus = summary[(summary["status"] == "ok") & (summary["cost_bps_per_side"] == 10.0)]
    passed = focus[focus["passes_lockbox_10bps"]]
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {YEARLY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"10 bps rows tested: {len(focus)}")
    print(f"Passed canonical gate: {len(passed)}")
    if len(passed):
        print(passed[["name", "ticker", "ema_period", "trail_atr", "lockbox_total_r"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
