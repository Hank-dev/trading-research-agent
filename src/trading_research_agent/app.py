import argparse
from datetime import UTC, datetime
import uuid
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_research_agent.reports.markdown_report import save_markdown_report
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.history import (
    append_run_record,
    format_summary,
    load_history,
    record_from_state,
    summarize_history,
)
from trading_research_agent.workflows.campaign_research import (
    CampaignResult,
    run_campaign,
)
from trading_research_agent.workflows.explore_research import (
    ExploreResult,
    run_exploration,
)
from trading_research_agent.workflows.portfolio_research import (
    run_portfolio_exploration,
    run_portfolio_spec,
)
from trading_research_agent.workflows.portfolio_batch import run_portfolio_batch
from trading_research_agent.workflows.iterative_research import (
    run_research_until_pass,
    run_research_with_one_iteration,
)
from trading_research_agent.workflows.research_graph import build_research_graph


def main(argv: list[str] | None = None) -> int:
    # Load .env up front so every mode (including --stress, which never touches an
    # LLM) sees TIINGO_API_KEY and other settings, not just the LLM-driven paths.
    load_dotenv()

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
        help="Save the generated Markdown report under outputs/",
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
        "--assets",
        type=str,
        default="",
        metavar="SYM1,SYM2,...",
        help="Comma-separated portfolio assets for --portfolio-spec or --data-health.",
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
        "--report-html",
        action="store_true",
        help=(
            "Render a self-contained, read-only HTML dashboard of history, verdicts, "
            "and forward paper-trade curves to outputs/dashboard.html. Exits after."
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
    args = parser.parse_args(argv)

    console = Console()

    if args.data_health:
        return _run_data_health_mode(console, args, parser)

    if args.combined_book:
        return _run_combined_book_mode(console, args, parser)

    if args.paper_trade:
        return _run_paper_trade_mode(console, inception=args.inception or None)

    if args.paper_status:
        return _run_paper_status_mode(console)

    if args.report_html:
        from trading_research_agent.reports.html_dashboard import build_html_report

        try:
            path = build_html_report()
        except Exception as exc:
            console.print(Panel(f"Could not build dashboard: {exc}", title="HTML Dashboard"))
            return 1
        console.print(Panel(f"Dashboard written to {path}\nOpen it in any browser.", title="HTML Dashboard"))
        return 0

    if args.stress:
        return _run_stress_mode(console, suggest=args.suggest)

    if args.history_detail:
        if args.suggest:
            parser.error("--suggest cannot be combined with --history-detail")
        return _run_history_detail_mode(console, args)

    if args.history:
        summary = summarize_history(load_history())
        console.print(Panel(format_summary(summary), title="Research History"))
        if args.suggest:
            _print_history_suggestion(console, summary)
        return 0

    if args.suggest:
        parser.error("--suggest must be used together with --history")

    if args.portfolio_batch:
        return _run_portfolio_batch_mode(console, args, parser)

    if args.portfolio_spec:
        return _run_portfolio_spec_mode(console, args, parser)

    if not args.idea:
        parser.error("idea is required unless --history is used")

    if args.portfolio:
        if args.explore <= 0:
            parser.error("--portfolio requires --explore N (e.g. --explore 4)")
        result = run_portfolio_exploration(
            args.idea,
            slate_size=args.explore,
            lockbox_pct=args.lockbox_pct,
        )
        _save_exploration_reports(result, args.save_report)
        _log_exploration_history(args.idea, result, mode="portfolio")
        _print_exploration_result(console, result)
        _refresh_dashboard_safely(console)
        return 1 if result.get("errors") else 0

    if args.campaign:
        if args.explore <= 0:
            parser.error("--campaign requires --explore N (e.g. --explore 3)")
        universe = [s for s in (sym.strip() for sym in args.universe.split(",")) if s]
        if not universe:
            parser.error("--campaign requires --universe SYM1,SYM2,...")
        result = run_campaign(
            idea=args.idea,
            universe=universe,
            slate_size=args.explore,
            lockbox_pct=args.lockbox_pct,
        )
        _save_campaign_reports(result, args.save_report)
        _log_campaign_history(args.idea, result)
        _print_campaign_result(console, result)
        _refresh_dashboard_safely(console)
        had_errors = any(slot.get("error") for slot in result.get("slots", []))
        return 1 if had_errors else 0

    if args.explore > 0:
        result = run_exploration(
            args.idea,
            slate_size=args.explore,
            lockbox_pct=args.lockbox_pct,
        )
        _save_exploration_reports(result, args.save_report)
        _log_exploration_history(args.idea, result)
        _print_exploration_result(console, result)
        _refresh_dashboard_safely(console)
        return 1 if result.get("errors") else 0
    if args.iterate_until_pass:
        result = run_research_until_pass(args.idea, max_iterations=args.max_iterations)
        _save_iterative_reports(result, args.save_report)
        _log_iterative_history(args.idea, result, mode="iterate-until-pass")
        _print_iterative_result(console, result)
        _refresh_dashboard_safely(console)
        return 1 if result.get("errors") else 0
    if args.iterate_once:
        result = run_research_with_one_iteration(args.idea)
        _save_iterative_reports(result, args.save_report)
        _log_iterative_history(args.idea, result, mode="iterate-once")
        _print_iterative_result(console, result)
        _refresh_dashboard_safely(console)
        return 1 if result.get("errors") else 0

    state = build_research_graph().invoke({"user_request": args.idea})
    _log_history_safely(record_from_state(state, mode="single", user_request=args.idea))

    if args.save_report and state.get("report"):
        strategy_name = getattr(state.get("strategy_spec"), "name", "research")
        state["report"] = save_markdown_report(state["report"], strategy_name)

    _print_state(console, state)
    _refresh_dashboard_safely(console)
    return 1 if state.get("errors") else 0


def _run_portfolio_spec_mode(
    console: Console, args: Any, parser: argparse.ArgumentParser
) -> int:
    if args.portfolio or args.campaign or args.portfolio_batch:
        parser.error("--portfolio-spec cannot be combined with --portfolio, --portfolio-batch or --campaign")
    if args.explore > 0:
        parser.error("--portfolio-spec runs one exact strategy and does not use --explore")

    assets = _parse_symbol_csv(args.assets)
    if not assets or not args.start or not args.end:
        parser.error("--portfolio-spec requires --assets, --start and --end")

    family = PortfolioFamily(args.family)
    if family == PortfolioFamily.CRISIS_HEDGE and args.hedge_weight is None:
        parser.error("--family crisis_hedge requires --hedge-weight")

    try:
        spec = PortfolioSpec(
            name=args.strategy_name or _default_portfolio_name(family, assets),
            assets=assets,
            portfolio_family=family,
            start_date=args.start,
            end_date=args.end,
            lookback_days=args.lookback,
            top_k=args.top_k,
            rebalance_days=args.rebalance,
            hedge_weight=(
                args.hedge_weight if family == PortfolioFamily.CRISIS_HEDGE else None
            ),
            hypothesis=(
                args.hypothesis
                or args.idea
                or _default_portfolio_hypothesis(family, assets)
            ),
        )
    except Exception as exc:
        parser.error(f"invalid --portfolio-spec: {exc}")

    user_request = args.idea or _portfolio_spec_request(spec)
    try:
        result = run_portfolio_spec(spec, user_request, lockbox_pct=args.lockbox_pct)
    except Exception as exc:
        console.print(Panel(f"Portfolio spec failed: {exc}", title="Portfolio Spec"))
        return 1

    _save_exploration_reports(result, args.save_report)
    slate_id = _log_exploration_history(user_request, result, mode="portfolio")
    _print_history_detail_hint(console, slate_id)
    _print_exploration_result(console, result)
    _refresh_dashboard_safely(console)
    return 1 if result.get("errors") else 0


def _run_portfolio_batch_mode(
    console: Console, args: Any, parser: argparse.ArgumentParser
) -> int:
    if args.portfolio or args.portfolio_spec or args.campaign:
        parser.error("--portfolio-batch cannot be combined with --portfolio, --portfolio-spec or --campaign")
    if args.explore > 0:
        parser.error("--portfolio-batch runs exact specs and does not use --explore")
    if args.idea:
        parser.error("--portfolio-batch does not take an idea; put hypotheses in the batch file")

    try:
        batch = run_portfolio_batch(args.portfolio_batch, lockbox_pct=args.lockbox_pct)
    except Exception as exc:
        console.print(Panel(f"Portfolio batch failed: {exc}", title="Portfolio Batch"))
        return 1

    lockbox_label = (
        f"{batch['lockbox_pct']:.0%} of date range"
        if batch["lockbox_pct"] > 0
        else "disabled"
    )
    header = [
        f"Spec file: {batch['path']}",
        f"Portfolios requested: {batch['count']}",
        f"Lockbox: {lockbox_label}",
    ]
    console.print(Panel("\n".join(header), title="Portfolio Batch"))

    had_errors = bool(batch.get("errors"))
    for entry in batch.get("results", []):
        spec = entry["spec"]
        result = entry["result"]
        console.rule(f"Batch portfolio {entry['index']}: {spec.name}")
        _save_exploration_reports(result, args.save_report)
        slate_id = _log_exploration_history(entry["user_request"], result, mode="portfolio")
        _print_history_detail_hint(console, slate_id)
        _print_exploration_result(console, result)
        had_errors = had_errors or bool(result.get("errors"))

    if batch.get("errors"):
        console.print(Panel("\n".join(batch["errors"]), title="Portfolio Batch Errors"))

    _refresh_dashboard_safely(console)
    return 1 if had_errors else 0


def _parse_symbol_csv(raw: str) -> list[str]:
    return [s for s in (sym.strip() for sym in raw.split(",")) if s]


def _default_portfolio_name(family: PortfolioFamily, assets: list[str]) -> str:
    family_name = family.value.replace("_", " ")
    return f"{family_name.title()} ({', '.join(assets)})"


def _default_portfolio_hypothesis(family: PortfolioFamily, assets: list[str]) -> str:
    if family == PortfolioFamily.CROSS_SECTIONAL_MOMENTUM:
        return (
            "Relative strength persists across major assets, so the portfolio rotates "
            "into the strongest trailing performers."
        )
    if family == PortfolioFamily.DUAL_MOMENTUM:
        return (
            "Relative strength persists, but assets with negative absolute momentum "
            "should leave their allocation in cash."
        )
    return (
        f"The {family.value} rule may improve risk-adjusted returns across "
        f"{', '.join(assets)}."
    )


def _portfolio_spec_request(spec: PortfolioSpec) -> str:
    return (
        f"Hand-specified portfolio: {spec.portfolio_family.value} across "
        f"{', '.join(spec.assets)} from {spec.start_date} to {spec.end_date}; "
        f"lookback={spec.lookback_days}, top_k={spec.top_k}, "
        f"rebalance={spec.rebalance_days}."
    )


def _run_data_health_mode(
    console: Console, args: Any, parser: argparse.ArgumentParser
) -> int:
    if args.idea:
        parser.error("--data-health does not take an idea; use --assets, --start, --end")
    assets = _parse_symbol_csv(args.assets)
    if not assets or not args.start or not args.end:
        parser.error("--data-health requires --assets, --start and --end")

    from trading_research_agent.workflows.data_health import check_data_health

    try:
        result = check_data_health(assets, args.start, args.end)
    except Exception as exc:
        console.print(Panel(f"Data health check failed: {exc}", title="Data Health"))
        return 1

    _print_data_health_result(console, result)
    return 0 if result["runnable"] else 1


def _run_history_detail_mode(console: Console, args: Any) -> int:
    from trading_research_agent.workflows.history_detail import (
        build_history_detail,
        render_history_detail_markdown,
        save_history_detail_report,
    )

    detail = build_history_detail(args.history_detail)
    markdown = render_history_detail_markdown(detail)
    console.print(Panel(markdown, title="Research History Detail"))

    if args.save_report and detail.get("status") == "ok":
        try:
            path = save_history_detail_report(markdown, args.history_detail)
        except OSError as exc:
            console.print(Panel(f"Could not save history-detail report: {exc}", title="History Detail"))
            return 1
        console.print(Panel(f"Report path: {path}", title="History Detail"))

    return 0 if detail.get("status") == "ok" else 1


def _print_data_health_result(console: Console, result: dict[str, Any]) -> None:
    table = Table(title="Asset Data Health")
    table.add_column("Asset")
    table.add_column("Source")
    table.add_column("Cache")
    table.add_column("Rows", justify="right")
    table.add_column("Range")
    table.add_column("Status")
    table.add_column("Detail")

    for check in result["checks"]:
        date_range = (
            f"{check.first_date} -> {check.last_date}"
            if check.first_date and check.last_date
            else "-"
        )
        status = "OK" if check.status == "ok" else "ERROR"
        table.add_row(
            check.asset,
            check.source,
            check.cache,
            str(check.rows),
            date_range,
            status,
            check.detail,
        )
    console.print(table)

    summary = [
        f"Requested range: {result['start']} to {result['end']}",
        f"Common aligned rows: {result['common_rows']}",
    ]
    if result.get("common_start") and result.get("common_end"):
        summary.append(
            f"Common aligned range: {result['common_start']} to {result['common_end']}"
        )
    summary.extend(
        [
            f"Minimum required rows: {result['min_rows']}",
            f"Portfolio runnable: {'YES' if result['runnable'] else 'NO'}",
            f"Reason: {result['reason']}",
        ]
    )
    console.print(Panel("\n".join(summary), title="Data Health Summary"))


def _run_combined_book_mode(console: Console, args: Any, parser: argparse.ArgumentParser) -> int:
    from trading_research_agent.schemas.combined_book import CombinedBookSpec
    from trading_research_agent.workflows.combined_book import (
        run_combined_book_eval,
        run_combined_book_with_lockbox,
    )

    core = [s for s in (x.strip() for x in args.core.split(",")) if s]
    overlay = [s for s in (x.strip() for x in args.overlay.split(",")) if s]
    if not core or not overlay or not args.start or not args.end:
        parser.error("--combined-book requires --core, --overlay, --start and --end")

    try:
        spec = CombinedBookSpec(
            core_assets=core,
            overlay_assets=overlay,
            overlay_weight=args.overlay_weight,
            overlay_rule=args.overlay_rule,
            start_date=args.start,
            end_date=args.end,
        )
        if args.lockbox_pct > 0:
            paired = run_combined_book_with_lockbox(spec, lockbox_pct=args.lockbox_pct)
        else:
            result = run_combined_book_eval(spec)
    except Exception as exc:
        console.print(Panel(f"Combined-book evaluation failed: {exc}", title="Combined Book"))
        return 1

    if args.lockbox_pct > 0:
        _print_combined_book_lockbox(console, paired)
    else:
        _print_combined_book_result(console, result)
    return 0


def _print_combined_book_lockbox(console: Console, paired: dict[str, Any]) -> None:
    split = paired.get("lockbox_split", {})
    console.print(
        Panel(
            "\n".join(
                [
                    f"Train segment:    {split.get('original_start')} to {split.get('train_end')}",
                    f"Held-out lockbox: {split.get('lockbox_start')} to {split.get('original_end')}",
                    "The overlay's benefit is judged on the held-out tail it never saw.",
                ]
            ),
            title="Combined Book — lockbox split",
        )
    )

    train = paired.get("train")
    lockbox = paired.get("lockbox")
    if train is not None:
        console.rule("Train segment")
        _print_combined_book_result(console, train)
    if lockbox is not None:
        console.rule("Held-out lockbox")
        _print_combined_book_result(console, lockbox)
    else:
        console.print(Panel("Held-out segment could not be evaluated.", title="Held-out lockbox"))

    train_verdict = train["comparison"]["verdict"] if train else "unavailable"
    lockbox_verdict = lockbox["comparison"]["verdict"] if lockbox else "unavailable"
    confirmed = paired.get("confirmed", False)
    console.print(
        Panel(
            "\n".join(
                [
                    f"Train verdict:     {train_verdict}",
                    f"Held-out verdict:  {lockbox_verdict}",
                    "",
                    f"OVERLAY BENEFIT CONFIRMED OUT OF SAMPLE: {'YES' if confirmed else 'NO'}",
                    "",
                    "Trust the held-out verdict. If the benefit vanished out of sample, "
                    "the overlay's value was an artifact of the full-period window.",
                ]
            ),
            title="Combined Book — Final Verdict",
        )
    )


def _print_combined_book_result(console: Console, result: dict[str, Any]) -> None:
    core = result["core"]
    combined = result["combined"]
    cmp = result["comparison"]

    def _fmt(metrics: dict[str, Any]) -> str:
        sharpe = metrics["sharpe_ratio"]
        sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "n/a"
        return (
            f"return {metrics['total_return_pct']:7.1f}%   "
            f"maxDD {metrics['max_drawdown_pct']:6.1f}%   "
            f"Sharpe {sharpe_s}"
        )

    rule = result["overlay_rule"]
    lines = [
        f"Core:    {', '.join(result['core_assets'])}",
        f"Overlay: {', '.join(result['overlay_assets'])} "
        f"@ {result['overlay_weight']:.0%} ({rule})",
        "",
        f"Core alone:   {_fmt(core)}",
        f"Combined:     {_fmt(combined)}",
        "",
        f"Return cost of the hedge:     {cmp['return_cost_pct']:6.1f}%",
        f"Drawdown improvement:         {cmp['drawdown_improvement_pct']:6.1f}%  (positive = shallower)",
        f"Sharpe delta:                 {cmp['sharpe_delta']:+.3f}  (positive = better risk-adjusted)",
        "",
        f"VERDICT: {cmp['verdict']}",
        "",
        _combined_book_gloss(cmp["verdict"]),
    ]
    console.print(Panel("\n".join(lines), title="Combined Book — overlay vs core alone"))


def _combined_book_gloss(verdict: str) -> str:
    if verdict == "IMPROVES_RISK_ADJUSTED":
        return (
            "The overlay raised the combined Sharpe — a genuine improvement, not just "
            "insurance. Rare. Worth carrying, and worth confirming on a held-out window."
        )
    if verdict == "REDUCES_DRAWDOWN_AT_COST":
        return (
            "The overlay did NOT improve risk-adjusted return, but it materially cut "
            "drawdown. That is a real but costly trade: you pay return for tail "
            "protection. Whether it is worth it depends on your risk tolerance."
        )
    return (
        "The overlay dragged returns without enough drawdown benefit to justify it. "
        "As a hedge for this core over this period, it was not worth holding."
    )


def _run_paper_trade_mode(console: Console, *, inception: str | None) -> int:
    from trading_research_agent.workflows.paper_trading import open_paper_position
    from trading_research_agent.workflows.robustness_stress import (
        latest_confirmed_portfolio_winner,
    )

    winner = latest_confirmed_portfolio_winner(load_history())
    if winner is None:
        console.print(
            Panel(
                "No lockbox-confirmed portfolio winner in history to paper-trade. "
                "Run a confirmed --portfolio first.",
                title="Paper Trade",
            )
        )
        return 1

    try:
        record = open_paper_position(winner, inception=inception)
    except Exception as exc:
        console.print(Panel(f"Could not open paper position: {exc}", title="Paper Trade"))
        return 1

    exp = record["expectation"]
    console.print(
        Panel(
            "\n".join(
                [
                    f"Position id:  {record['id']}",
                    f"Strategy:     {record['strategy_family']}",
                    f"Universe:     {', '.join(record['params']['assets'])}",
                    f"Inception:    {record['inception_date']}  "
                    "(forward / out-of-sample from here)",
                    f"Expectation:  {exp['annualized_return_pct']:.1f}% annualized "
                    f"(backtest max drawdown {exp['backtest_max_drawdown_pct']:.1f}%)",
                    "",
                    "Paper position recorded. Run --paper-status over coming weeks/months "
                    "to see whether reality matches the backtest. This is the only test "
                    "that cannot be fit in hindsight.",
                ]
            ),
            title="Paper Trade — opened",
        )
    )
    _refresh_dashboard_safely(console)
    return 0


def _run_paper_status_mode(console: Console) -> int:
    from trading_research_agent.workflows.paper_trading import (
        evaluate_paper_position,
        load_paper_positions,
    )

    positions = [p for p in load_paper_positions() if p.get("status") == "open"]
    if not positions:
        console.print(
            Panel(
                "No open paper positions. Open one with --paper-trade.",
                title="Paper Status",
            )
        )
        _refresh_dashboard_safely(console)
        return 0

    for position in positions:
        ev = evaluate_paper_position(position)
        console.rule(f"Paper position {ev['id']}")
        if ev["status"] == "error":
            console.print(Panel(f"Could not evaluate: {ev['detail']}", title="Paper Status"))
            continue
        if ev["status"] == "no_data_yet":
            console.print(
                Panel(
                    f"Inception {ev['inception_date']} — {ev['detail']}",
                    title="Paper Status",
                )
            )
            continue

        lines = [
            f"Inception:            {ev['inception_date']}  ->  {ev['as_of']}",
            f"Forward trading days: {ev['forward_trading_days']}",
            "",
            f"Realized return:      {ev['realized_return_pct']:7.1f}%",
            f"Realized annualized:  {ev['realized_annualized_pct']:7.1f}%   "
            f"(backtest expected {ev['expected_annualized_pct']:.1f}%)",
            f"Realized max DD:      {ev['realized_max_drawdown_pct']:7.1f}%   "
            f"(backtest worst {ev['backtest_max_drawdown_pct']:.1f}%)",
            "",
            f"READ: {ev['read']}",
            "",
            ev["detail"],
        ]
        console.print(Panel("\n".join(lines), title="Paper Status — forward vs backtest"))
    _refresh_dashboard_safely(console)
    return 0


def _run_stress_mode(console: Console, *, suggest: bool) -> int:
    from trading_research_agent.workflows.robustness_stress import (
        latest_confirmed_portfolio_winner,
        run_stress_test,
        spec_from_winner,
    )

    winner = latest_confirmed_portfolio_winner(load_history())
    if winner is None:
        console.print(
            Panel(
                "No lockbox-confirmed portfolio winner found in history. Run a "
                "confirmed portfolio first:\n"
                "  trade-research --portfolio --explore 4 --lockbox-pct 0.2 \"...\"",
                title="Robustness Stress Test",
            )
        )
        return 1

    spec = spec_from_winner(winner)
    console.print(
        Panel(
            "\n".join(
                [
                    f"Strategy:  {spec.portfolio_family.value}",
                    f"Universe:  {', '.join(spec.assets)}",
                    f"Params:    lookback={spec.lookback_days}, top_k={spec.top_k}, "
                    f"rebalance={spec.rebalance_days}",
                    f"Range:     {spec.start_date} to {spec.end_date}",
                    "",
                    "Perturbing lockbox cuts, parameter neighbors, and leave-one-out "
                    "universes. This runs many backtests and may take a minute.",
                ]
            ),
            title="Robustness Stress Test — target",
        )
    )

    try:
        stress = run_stress_test(spec, spec.start_date, spec.end_date)
    except Exception as exc:
        console.print(Panel(f"Stress test failed: {exc}", title="Robustness Stress Test"))
        return 1

    _print_stress_result(console, stress)
    _log_stress_history(spec, stress)

    if suggest:
        _print_stress_interpretation(console, stress)

    _refresh_dashboard_safely(console)
    return 0


def _print_stress_result(console: Console, stress: dict[str, Any]) -> None:
    summary = stress.get("summary", {})
    lines: list[str] = []
    for r in stress.get("results", []):
        if r.get("status") != "ok":
            lines.append(f"  [{r.get('category'):<9}] {r.get('label'):<22} UNRUNNABLE")
            continue
        mark = "CONFIRMS" if r.get("confirms") else "fails   "
        ret = r.get("held_out_return_pct")
        bench = r.get("held_out_benchmark_pct")
        ret_s = f"{ret:6.1f}%" if ret is not None else "   n/a"
        bench_s = f"{bench:6.1f}%" if bench is not None else "   n/a"
        lines.append(
            f"  [{r.get('category'):<9}] {r.get('label'):<22} {mark} "
            f"held-out {ret_s} vs bench {bench_s}"
        )

    cat_lines = []
    for category, stats in summary.get("category_rates", {}).items():
        cat_lines.append(
            f"  {category:<9} {stats['confirmed']}/{stats['total']} confirm ({stats['rate']:.0%})"
        )

    verdict = summary.get("verdict", "?")
    overall = (
        f"{summary.get('overall_confirmed', 0)}/{summary.get('overall_runnable', 0)} "
        f"({summary.get('overall_rate', 0):.0%})"
    )
    body = (
        "\n".join(lines)
        + "\n\nSurvival by category:\n"
        + "\n".join(cat_lines)
        + f"\n\nOverall held-out survival: {overall}"
        + f"\n\nVERDICT: {verdict}"
        + "\n\n"
        + _stress_verdict_gloss(verdict)
    )
    console.print(Panel(body, title="Robustness Stress Test — results"))


def _log_stress_history(spec: PortfolioSpec, stress: dict[str, Any]) -> None:
    summary = stress.get("summary", {})
    record = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "stress",
        "user_request": "Robustness stress test of the latest confirmed portfolio winner.",
        "is_lockbox": False,
        "asset": "PORTFOLIO[" + ",".join(spec.assets) + "]",
        "strategy_family": spec.portfolio_family.value,
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "params": {
            "assets": list(spec.assets),
            "lookback_days": spec.lookback_days,
            "top_k": spec.top_k,
            "rebalance_days": spec.rebalance_days,
        },
        "verdict": summary.get("verdict", "unknown"),
        "stress_summary": summary,
    }
    _log_history_safely(record)


def _stress_verdict_gloss(verdict: str) -> str:
    if verdict == "ROBUST":
        return (
            "The edge survives its own neighborhood: nearby parameters, different "
            "held-out windows, and dropping any single asset. This is the strongest "
            "evidence this toolkit can produce. Next step is forward paper trading, "
            "not more in-sample work."
        )
    if verdict == "FRAGILE":
        return (
            "The original pass held, but the edge does NOT survive perturbation well "
            "— it is closer to a knife-edge than a robust effect. Distrust it. Do not "
            "tune it to 'fix' the failures; that is the overfitting trap."
        )
    return (
        "The strategy did not even confirm at its baseline cut under re-test. Treat "
        "the original pass as noise."
    )


def _print_stress_interpretation(console: Console, stress: dict[str, Any]) -> None:
    try:
        from trading_research_agent.nodes.interpret_robustness import interpret_robustness

        interp = interpret_robustness(stress)
    except Exception as exc:
        console.print(Panel(f"Could not get LLM interpretation: {exc}", title="LLM Read (Grok)"))
        return

    lines = [interp.assessment, ""]
    if interp.fragility_flags:
        lines.append("Fragility flags:")
        lines.extend(f"  - {flag}" for flag in interp.fragility_flags)
        lines.append("")
    lines.append(f"Recommendation: {interp.recommendation}")
    console.print(Panel("\n".join(lines), title="LLM Read (Grok)"))


def _print_history_suggestion(console: Console, summary: dict[str, Any]) -> None:
    if summary.get("total_trials", 0) == 0:
        console.print(
            Panel(
                "No history to analyze yet — run some backtests first.",
                title="LLM Suggestion",
            )
        )
        return

    try:
        from trading_research_agent.nodes.suggest_history_directions import (
            suggest_directions_from_history,
        )

        suggestion = suggest_directions_from_history(summary)
    except Exception as exc:
        console.print(Panel(f"Could not get LLM suggestion: {exc}", title="LLM Suggestion"))
        return

    lines: list[str] = [suggestion.summary, ""]
    if suggestion.structural_gaps:
        lines.append("Structural gaps (things you have NOT tried):")
        for gap in suggestion.structural_gaps:
            lines.append(f"  - {gap}")
        lines.append("")
    if suggestion.next_directions:
        lines.append("Next directions (structural, not parameter tweaks):")
        for direction in suggestion.next_directions:
            lines.append(f"  - {direction}")
        lines.append("")
    if suggestion.honest_warnings:
        lines.append("Honest warnings:")
        for warning in suggestion.honest_warnings:
            lines.append(f"  - {warning}")

    console.print(Panel("\n".join(lines).rstrip(), title="LLM Suggestion (Grok)"))


def _log_history_safely(record: dict[str, Any] | None) -> None:
    """History is a nice-to-have. Don't ever fail a run because we couldn't write to it."""
    if record is None:
        return
    try:
        append_run_record(record)
    except OSError:
        pass


def _refresh_dashboard_safely(console: Console) -> None:
    """Refresh the static dashboard without letting HTML generation fail the run."""
    try:
        from trading_research_agent.reports.html_dashboard import build_html_report

        path = build_html_report()
    except Exception as exc:
        console.print(
            Panel(
                f"Backtest finished, but dashboard refresh failed: {exc}",
                title="HTML Dashboard",
            )
        )
        return

    console.print(Panel(f"Dashboard refreshed: {path}", title="HTML Dashboard"))


def _print_history_detail_hint(console: Console, slate_id: str | None) -> None:
    if not slate_id:
        return
    console.print(
        Panel(
            f"Run id: {slate_id}\nDetails: trade-research --history-detail {slate_id}",
            title="History",
        )
    )


def _log_exploration_history(
    user_request: str, result: ExploreResult, *, mode: str = "explore"
) -> str:
    slate_id = uuid.uuid4().hex[:8]
    for candidate in result.get("candidates", []):
        _log_history_safely(
            record_from_state(
                candidate,
                mode=mode,
                user_request=user_request,
                slate_id=slate_id,
            )
        )
    lockbox = result.get("lockbox")
    if lockbox:
        _log_history_safely(
            record_from_state(
                lockbox,
                mode=mode,
                user_request=user_request,
                slate_id=slate_id,
                is_lockbox=True,
            )
        )
    return slate_id


def _log_iterative_history(user_request: str, result: dict[str, Any], *, mode: str) -> None:
    initial = result.get("initial")
    if initial:
        _log_history_safely(record_from_state(initial, mode=mode, user_request=user_request))
    for state in result.get("iterations", []):
        _log_history_safely(record_from_state(state, mode=mode, user_request=user_request))


def _save_campaign_reports(result: CampaignResult, save_report: bool) -> None:
    if not save_report:
        return
    for slot in result.get("slots", []):
        exploration = slot.get("exploration")
        if exploration is None:
            continue
        _save_exploration_reports(exploration, save_report=True)


def _log_campaign_history(idea: str, result: CampaignResult) -> None:
    for slot in result.get("slots", []):
        exploration = slot.get("exploration")
        if exploration is None:
            continue
        per_asset_request = f"{idea} on {slot['asset']}"
        _log_exploration_history(per_asset_request, exploration)


def _print_campaign_result(console: Console, result: CampaignResult) -> None:
    universe = result.get("universe", [])
    slate_size = result.get("slate_size", 0)
    lockbox_pct = result.get("lockbox_pct", 0.0)
    summary = result.get("summary") or {}

    header = [
        f"Idea: {result.get('idea', '')}",
        f"Universe ({len(universe)}): {', '.join(universe)}",
        f"Per-asset slate size: {slate_size}",
        f"Lockbox: {f'{lockbox_pct:.0%} of date range' if lockbox_pct > 0 else 'disabled'}",
        f"Total trials this campaign: {summary.get('total_trials', 0)} "
        f"({summary.get('total_lockbox_runs', 0)} lockbox verification(s))",
    ]
    console.print(Panel("\n".join(header), title="Campaign"))

    for slot in result.get("slots", []):
        asset = slot.get("asset", "?")
        if slot.get("error"):
            console.rule(f"Asset: {asset} (error)")
            console.print(Panel(slot["error"], title=f"Error on {asset}"))
            continue

        console.rule(f"Asset: {asset}")
        exploration = slot.get("exploration")
        if exploration is not None:
            _print_exploration_result(console, exploration)

    _print_campaign_summary(console, result, summary)


def _print_campaign_summary(
    console: Console, result: CampaignResult, summary: dict[str, Any]
) -> None:
    universe = result.get("universe", [])
    if not universe:
        return

    lines: list[str] = []
    lines.append("Per-asset outcome:")
    for slot in result.get("slots", []):
        asset = slot.get("asset", "?")
        if slot.get("error"):
            lines.append(f"  {asset:<10} ERROR: {slot['error']}")
            continue
        pass_count = slot.get("pass_count", 0)
        lockbox = slot.get("lockbox_verdict") or "n/a"
        lines.append(
            f"  {asset:<10} slate pass: {pass_count}/{result.get('slate_size', 0)}    "
            f"lockbox verdict: {lockbox}"
        )

    assets_pass = summary.get("assets_with_any_pass", [])
    lockbox_pass = summary.get("assets_with_lockbox_pass", [])
    lines.append("")
    lines.append(
        f"Assets with any slate pass:    {', '.join(assets_pass) if assets_pass else 'none'}"
    )
    lines.append(
        f"Assets with lockbox pass:      {', '.join(lockbox_pass) if lockbox_pass else 'none'}"
    )

    failed_checks = summary.get("failed_checks", {})
    if failed_checks:
        lines.append("")
        lines.append("Most-failed robustness checks across campaign:")
        for check, count in failed_checks.items():
            lines.append(f"  {count:>4}x  {check}")

    lines.append("")
    lines.append("Cross-run note:")
    lines.append(
        f"  Campaign trial count: {summary.get('total_trials', 0)}. Each per-asset slate's"
    )
    lines.append(
        "  DSR was computed within that slate only. Across the campaign, the effective"
    )
    lines.append(
        "  multiple-testing penalty is larger. Use --history --suggest to read this honestly."
    )

    console.print(Panel("\n".join(lines), title="Campaign Summary"))


def _save_exploration_reports(result: ExploreResult, save_report: bool) -> None:
    if not save_report:
        return
    for index, candidate in enumerate(result.get("candidates", []), start=1):
        if not candidate.get("report"):
            continue
        strategy_name = getattr(candidate.get("strategy_spec"), "name", f"candidate_{index}")
        candidate["report"] = save_markdown_report(
            candidate["report"], f"candidate_{index}_{strategy_name}"
        )
    lockbox = result.get("lockbox")
    if lockbox and lockbox.get("report"):
        strategy_name = getattr(lockbox.get("strategy_spec"), "name", "lockbox")
        lockbox["report"] = save_markdown_report(lockbox["report"], f"lockbox_{strategy_name}")


def _print_exploration_result(console: Console, result: ExploreResult) -> None:
    candidates = result.get("candidates", [])
    winner_index = result.get("winner_index")
    winner_reason = result.get("winner_reason", "")
    errors = result.get("errors", [])
    lockbox_split = result.get("lockbox_split")

    if lockbox_split:
        console.print(
            Panel(
                "\n".join(
                    [
                        f"Original range: {lockbox_split['original_start']} to "
                        f"{lockbox_split['original_end']}",
                        f"Train/validation: {lockbox_split['original_start']} to "
                        f"{lockbox_split['train_end']}",
                        f"Held-out lockbox: {lockbox_split['lockbox_start']} to "
                        f"{lockbox_split['original_end']}",
                        "Slate generation only saw the train/validation segment.",
                    ]
                ),
                title="Lockbox Split",
            )
        )

    for index, candidate in enumerate(candidates, start=1):
        marker = " (winner)" if winner_index is not None and index - 1 == winner_index else ""
        console.rule(f"Candidate {index}{marker}")
        _print_state(console, candidate)

    summary = [
        f"Slate size: {len(candidates)}",
        f"Winner: candidate {winner_index + 1 if winner_index is not None else 'none'}",
        f"Selection rule: {winner_reason}",
    ]
    console.print(Panel("\n".join(summary), title="Exploration Summary"))

    failure_summary = result.get("failure_summary")
    if failure_summary:
        _print_failure_diagnostic(console, failure_summary)

    lockbox = result.get("lockbox")
    if lockbox:
        console.rule("Held-out lockbox re-test")
        _print_state(console, lockbox)

    _print_final_verdict(console, result)

    if errors:
        console.print(Panel("\n".join(errors), title="Exploration Errors"))


def _verdict_of_state(state: dict[str, Any] | None) -> str | None:
    if state is None:
        return None
    report = state.get("report")
    return report.verdict if report is not None else None


def _print_final_verdict(console: Console, result: ExploreResult) -> None:
    winner_index = result.get("winner_index")
    candidates = result.get("candidates", [])
    if winner_index is None or winner_index >= len(candidates):
        console.print(
            Panel(
                "No winner: no candidate produced a usable backtest.",
                title="Final Verdict",
            )
        )
        return

    winner_verdict = _verdict_of_state(candidates[winner_index]) or "unavailable"
    lockbox = result.get("lockbox")

    if lockbox is not None:
        lockbox_verdict = _verdict_of_state(lockbox) or "unavailable"
        lockbox_errors = lockbox.get("errors", [])
        if lockbox_errors or lockbox.get("backtest_result") is None:
            lines = [
                f"Winner in-slate verdict:   {winner_verdict}",
                f"Held-out lockbox verdict:  {lockbox_verdict}",
                "",
                "CONFIRMED ON HELD-OUT DATA: NO",
                "",
                "The lockbox did not produce valid backtest evidence. Fix the "
                "data/source error and rerun; do not treat this as a strategy "
                "failure.",
            ]
            console.print(Panel("\n".join(lines), title="Final Verdict"))
            return

        # The lockbox is the GATE: a strategy is only confirmed if it survives the
        # held-out segment the slate never saw. The in-slate verdict alone is not
        # enough, no matter how good it looks.
        confirmed = lockbox_verdict == "worth_paper_trading"
        lines = [
            f"Winner in-slate verdict:   {winner_verdict}",
            f"Held-out lockbox verdict:  {lockbox_verdict}",
            "",
            f"CONFIRMED ON HELD-OUT DATA: {'YES' if confirmed else 'NO'}",
            "",
            "The lockbox verdict is the one to trust — it is the only segment "
            "neither slate generation nor selection ever saw.",
        ]
        if confirmed:
            lines.extend(
                [
                    "",
                    "Next robustness gate: trade-research --stress",
                    "If stress survives, then use trade-research --paper-trade for forward evidence.",
                ]
            )
    else:
        lines = [
            f"Winner verdict: {winner_verdict}",
            "",
            "No held-out lockbox was run. Add --lockbox-pct 0.2 to confirm on data "
            "the slate never saw. Until then, treat this as in-sample-adjacent, not "
            "confirmed.",
        ]

    console.print(Panel("\n".join(lines), title="Final Verdict"))


def _print_failure_diagnostic(console: Console, summary: dict[str, Any]) -> None:
    verdict_counts = summary.get("verdict_counts", {})
    failed_check_counts = summary.get("failed_check_counts", {})
    with_backtest = summary.get("candidates_with_backtest", 0)
    without_backtest = summary.get("candidates_without_backtest", 0)

    lines: list[str] = []
    if verdict_counts:
        lines.append("Verdicts across slate:")
        for verdict in ("worth_paper_trading", "needs_more_testing", "reject"):
            if verdict in verdict_counts:
                lines.append(f"  {verdict_counts[verdict]}x  {verdict}")
        for verdict, count in verdict_counts.items():
            if verdict not in {"worth_paper_trading", "needs_more_testing", "reject"}:
                lines.append(f"  {count}x  {verdict}")
        lines.append("")

    if failed_check_counts:
        lines.append(f"Failed robustness checks across {with_backtest} backtested candidate(s):")
        for name, count in failed_check_counts.items():
            lines.append(f"  {count}x  {name}")
    else:
        lines.append("No failed robustness checks recorded.")

    if without_backtest:
        lines.append("")
        lines.append(f"{without_backtest} candidate(s) produced no backtest (excluded from check tally).")

    console.print(Panel("\n".join(lines), title="Failure Diagnostic"))


def _save_iterative_reports(result: dict[str, Any], save_report: bool) -> None:
    if not save_report:
        return
    initial = result.get("initial")
    if initial and initial.get("report"):
        strategy_name = getattr(initial.get("strategy_spec"), "name", "initial")
        initial["report"] = save_markdown_report(initial["report"], f"initial_{strategy_name}")

    for index, state in enumerate(result.get("iterations", []), start=1):
        if not state.get("report"):
            continue
        strategy_name = getattr(state.get("strategy_spec"), "name", f"iteration_{index}")
        state["report"] = save_markdown_report(
            state["report"], f"iteration_{index}_{strategy_name}"
        )


def _print_iterative_result(console: Console, result: dict[str, Any]) -> None:
    initial = result.get("initial")
    iterations = result.get("iterations", [])
    errors = result.get("errors", [])

    if initial:
        console.rule("Initial Backtest")
        _print_state(console, initial)
    for index, iteration in enumerate(iterations, start=1):
        console.rule(f"Iteration {index}")
        _print_state(console, iteration)
    if "passed" in result or "stop_reason" in result:
        status = [
            f"Passed: {result.get('passed', False)}",
            f"Stop reason: {result.get('stop_reason', 'unavailable')}",
            f"Follow-up iterations: {len(iterations)}",
        ]
        console.print(Panel("\n".join(status), title="Iteration Summary"))
    if errors:
        console.print(Panel("\n".join(errors), title="Iteration Errors"))


def _print_state(console: Console, state: dict[str, Any]) -> None:
    spec = state.get("strategy_spec")
    critique = state.get("critique")
    result = state.get("backtest_result")
    report = state.get("report")
    errors = state.get("errors", [])

    if spec:
        console.print(Panel(spec.model_dump_json(indent=2), title="Parsed Strategy Specification"))
    if critique:
        console.print(Panel(critique.model_dump_json(indent=2), title="Critique"))
    if result:
        console.print(Panel(result.metrics.model_dump_json(indent=2), title="Backtest Metrics"))
        checks = "\n".join(
            f"{check.test_name}: {'pass' if check.passed else 'fail'} - {check.details}"
            for check in result.robustness_results
        )
        console.print(Panel(checks or "No robustness checks ran.", title="Robustness Checks"))
    if report:
        console.print(Panel(report.verdict, title="Final Verdict"))
        if report.report_path:
            console.print(f"Report path: {report.report_path}")
    if errors:
        console.print(Panel("\n".join(errors), title="Errors"))


if __name__ == "__main__":
    raise SystemExit(main())
