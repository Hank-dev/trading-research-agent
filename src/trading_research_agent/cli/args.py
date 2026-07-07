"""Argument parser construction for the trade-research CLI.

Lifted verbatim from app.main so the parser definition lives apart from dispatch.
"""
from __future__ import annotations

import argparse

from trading_research_agent.schemas.portfolio import PortfolioFamily


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-research",
        description="Run a reproducible trading strategy research pipeline.",
    )
    parser.add_argument(
        "idea",
        nargs="?",
        default=None,
        help="Natural-language trading research idea (omit when using --history).",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save the generated Markdown report under the configured output directory.",
    )
    parser.add_argument(
        "--iterate-once",
        action="store_true",
        help="Run one LLM-generated follow-up strategy after the first backtest.",
    )
    parser.add_argument(
        "--iterate-until-pass",
        action="store_true",
        help="Iterate revised strategies until verdict is worth_paper_trading or the cap is reached.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum follow-up iterations for --iterate-until-pass. Default: 5.",
    )
    parser.add_argument(
        "--explore",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Pre-register a slate of N distinct strategies, backtest all, and report "
            "results side-by-side. No pass/fail loop."
        ),
    )
    parser.add_argument(
        "--lockbox-pct",
        type=float,
        default=0.0,
        metavar="X",
        help=(
            "Reserve the trailing fraction X (0 < X < 1) of the date range as a "
            "held-out test segment. Only the slate winner is re-run on it. "
            "Used with --explore and --portfolio-spec."
        ),
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help=(
            "Multi-asset portfolio mode: a Grok research-director pre-registers a "
            "slate of distinct rotation strategies (momentum / dual-momentum / "
            "trend), all backtested across an asset universe. Requires --explore N. "
            "Combine with --lockbox-pct for a held-out re-test of the winner."
        ),
    )
    parser.add_argument(
        "--portfolio-spec",
        action="store_true",
        help=(
            "Run one hand-specified multi-asset portfolio deterministically. "
            "Requires --assets, --family, --start and --end. Combine with "
            "--lockbox-pct for a held-out re-test."
        ),
    )
    parser.add_argument(
        "--portfolio-batch",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Run multiple exact portfolio specs from a JSON/YAML file. The file "
            "may contain a top-level `portfolios` list and optional `defaults`."
        ),
    )
    parser.add_argument(
        "--creative-lab",
        action="store_true",
        help=(
            "Run a creative but anti-overfit portfolio lab: pre-register a bounded "
            "slate of structurally different strategies, train-screen them once, "
            "lockbox every train survivor, then stress-test lockbox survivors. "
            "Requires --assets, --start, --end. Uses --explore N as candidate cap "
            "or defaults to 8."
        ),
    )
    parser.add_argument(
        "--research-slate",
        action="store_true",
        help=(
            "With --creative-lab, use the positional idea as a research goal: first "
            "write a research brief and falsifiable structural hypotheses, then "
            "convert them into the frozen pre-registered slate."
        ),
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="",
        metavar="SYM1,SYM2,...",
        help="Comma-separated portfolio assets for --portfolio-spec, --creative-lab or --data-health.",
    )
    parser.add_argument(
        "--family",
        choices=[family.value for family in PortfolioFamily],
        default=PortfolioFamily.CROSS_SECTIONAL_MOMENTUM.value,
        help="Portfolio family for --portfolio-spec. Default: cross_sectional_momentum.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        metavar="DAYS",
        help="Lookback window in trading days for --portfolio-spec. Default: 126.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        metavar="K",
        help="Number of top-ranked assets to hold for momentum families. Default: 1.",
    )
    parser.add_argument(
        "--rebalance",
        type=int,
        default=21,
        metavar="DAYS",
        help="Rebalance cadence in trading days for --portfolio-spec. Default: 21.",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="",
        metavar="NAME",
        help="Optional display name for --portfolio-spec.",
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        metavar="TEXT",
        help="Optional hypothesis text for --portfolio-spec.",
    )
    parser.add_argument(
        "--hedge-weight",
        type=float,
        default=None,
        metavar="W",
        help="crisis_hedge only: hedge allocation in (0, 0.5].",
    )
    parser.add_argument(
        "--campaign",
        action="store_true",
        help=(
            "Run the same idea across every asset in --universe as a sequence of "
            "--explore slates. Requires --universe and --explore."
        ),
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="",
        metavar="SYM1,SYM2,...",
        help="Comma-separated list of asset symbols for --campaign.",
    )
    parser.add_argument(
        "--combined-book",
        action="store_true",
        help=(
            "Evaluate whether a hedge overlay improves a core book vs holding the "
            "core alone. Requires --core, --overlay, --start, --end. The honest way "
            "to judge a crisis hedge, which loses money standalone but can still "
            "reduce a combined book's drawdown."
        ),
    )
    parser.add_argument("--core", type=str, default="", metavar="SYM1,SYM2", help="Core assets for --combined-book.")
    parser.add_argument("--overlay", type=str, default="", metavar="SYM1,SYM2", help="Overlay/hedge assets for --combined-book.")
    parser.add_argument("--overlay-weight", type=float, default=0.1, metavar="W", help="Overlay fraction (0,0.5]. Default 0.1.")
    parser.add_argument(
        "--overlay-rule",
        choices=["static", "regime"],
        default="regime",
        help="static = hold overlay always; regime = only when core trend is down. Default regime.",
    )
    parser.add_argument("--start", type=str, default="", metavar="YYYY-MM-DD", help="Start date for --combined-book, --portfolio-spec, or --data-health.")
    parser.add_argument("--end", type=str, default="", metavar="YYYY-MM-DD", help="End date for --combined-book, --portfolio-spec, or --data-health.")
    parser.add_argument(
        "--data-health",
        action="store_true",
        help=(
            "Preflight asset data availability, Tiingo cache coverage, quota/auth "
            "errors, and common aligned row count. Requires --assets, --start, --end."
        ),
    )
    parser.add_argument(
        "--mine-anomalies",
        action="store_true",
        help=(
            "Mine concrete empirical anomaly facts (lead/lag and regime-conditioned "
            "asymmetries) for a supplied asset universe. Requires --assets, --start, --end."
        ),
    )
    parser.add_argument(
        "--event-followthrough",
        action="store_true",
        help=(
            "Generate frozen delayed event-trigger strategies from mined event-followthrough "
            "facts, then evaluate train/lockbox/neighborhood stress. Requires --assets, "
            "--start, --end, --lockbox-pct. Uses --explore N as candidate cap or defaults to 6."
        ),
    )
    parser.add_argument(
        "--top-anomalies",
        type=int,
        default=12,
        metavar="N",
        help="Maximum anomaly facts to report for --mine-anomalies. Default: 12.",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help=(
            "Open a forward paper position on the most recent lockbox-confirmed "
            "portfolio winner. Inception defaults to the backtest end date (the "
            "out-of-sample boundary). Exits after recording."
        ),
    )
    parser.add_argument(
        "--inception",
        type=str,
        default="",
        metavar="YYYY-MM-DD",
        help="Optional inception date for --paper-trade (default: backtest end date).",
    )
    parser.add_argument(
        "--paper-status",
        action="store_true",
        help="Replay open paper positions forward from inception and report live performance vs the backtest.",
    )
    parser.add_argument(
        "--live-open",
        action="store_true",
        help=(
            "Open a stateful live paper book on the most recent lockbox-confirmed "
            "portfolio winner. Does not place broker orders."
        ),
    )
    parser.add_argument(
        "--live-tick",
        action="store_true",
        help=(
            "Advance stateful live paper book(s) through the latest available bar, "
            "persisting cash, positions, and the daily ledger."
        ),
    )
    parser.add_argument(
        "--live-status",
        action="store_true",
        help="List/evaluate stateful live paper book(s) without advancing them.",
    )
    parser.add_argument(
        "--live-auto-promote",
        action="store_true",
        help=(
            "Open a new parallel live paper book if history contains a newer "
            "lockbox-confirmed winner than the current open books."
        ),
    )
    parser.add_argument(
        "--book-id",
        type=str,
        default="",
        metavar="ID",
        help="Optional live paper book id for --live-tick or --live-status.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default="",
        metavar="YYYY-MM-DD",
        help="Optional cutoff date for --live-tick (default: today).",
    )
    parser.add_argument(
        "--report-html",
        action="store_true",
        help=(
            "Render a self-contained, read-only HTML dashboard of history, verdicts, "
            "and forward paper-trade curves to the configured output directory. Exits after."
        ),
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help=(
            "Robustness stress-test the most recent lockbox-confirmed portfolio "
            "winner from history: perturb lockbox cut points, parameter neighbors, "
            "and leave-one-out universes, and report how much of the edge survives. "
            "Exits after running."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print a cross-run summary of past backtests and exit.",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help=(
            "Print the cumulative trial budget: how many portfolio strategies you "
            "have tested, and whether any lockbox-confirmed candidate still clears "
            "the deflated-Sharpe bar once corrected for the full shot count."
        ),
    )
    parser.add_argument(
        "--macro-regime",
        type=str,
        default="",
        choices=["walcl", "fedfunds", "m2"],
        metavar="SIGNAL",
        help=(
            "Run the pre-registered macro-regime rotation on one monetary signal "
            "(walcl=Fed balance sheet, fedfunds=rate, m2=money stock). Frozen mapping, "
            "look-ahead-safe. Requires --start, --end, --lockbox-pct."
        ),
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="",
        metavar="PARAM",
        help=(
            "Sweep one single-asset parameter as a robustness test (PLATEAU vs SPIKE), "
            "not a peak-picker. e.g. --sweep entry_window. Requires --sweep-values, "
            "--sweep-asset, --sweep-family, --start, --end, --lockbox-pct."
        ),
    )
    parser.add_argument(
        "--sweep-values",
        type=str,
        default="",
        metavar="V1,V2,...",
        help="Comma-separated parameter values to sweep, e.g. 20,30,40,55,70,90.",
    )
    parser.add_argument(
        "--sweep-asset",
        type=str,
        default="",
        metavar="SYM",
        help="Single asset for the sweep, e.g. BTC-USD.",
    )
    parser.add_argument(
        "--sweep-family",
        type=str,
        default="donchian_breakout",
        choices=[
            "sma_crossover",
            "donchian_breakout",
            "rsi_mean_reversion",
            "filtered_donchian_breakout",
            "filtered_rsi_mean_reversion",
        ],
        help="Single-asset strategy family for the sweep. Default: donchian_breakout.",
    )
    parser.add_argument(
        "--history-detail",
        type=str,
        default="",
        metavar="ID_OR_QUERY",
        help=(
            "Print a readable report for a past run. Matches a slate id, asset "
            "universe, or strategy family query without rerunning the backtest."
        ),
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help=(
            "When combined with --history, ask the LLM for STRUCTURAL observations "
            "and at-most-three next directions based on the cross-run log. "
            "Does NOT suggest parameter tweaks of strategies already tried."
        ),
    )
    return parser
