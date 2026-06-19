import argparse
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from trading_research_agent.reports.markdown_report import save_markdown_report
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.history import (
    format_summary,
    load_history,
    record_from_state,
    summarize_history,
)
from trading_research_agent.workflows.campaign_research import (
    run_campaign,
)
from trading_research_agent.workflows.explore_research import (
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
from trading_research_agent.cli.args import build_parser
from trading_research_agent.cli.render import (
    _print_data_health_result,
    _print_combined_book_lockbox,
    _print_combined_book_result,
    _print_stress_result,
    _print_stress_interpretation,
    _print_history_suggestion,
    _print_history_detail_hint,
    _print_campaign_result,
    _print_exploration_result,
    _print_iterative_result,
    _print_state,
)
from trading_research_agent.cli.history_io import (
    _log_stress_history,
    _log_history_safely,
    _refresh_dashboard_safely,
    _log_exploration_history,
    _log_iterative_history,
    _save_campaign_reports,
    _log_campaign_history,
    _save_exploration_reports,
    _save_iterative_reports,
)


def main(argv: list[str] | None = None) -> int:
    # Load .env up front so every mode (including --stress, which never touches an
    # LLM) sees TIINGO_API_KEY and other settings, not just the LLM-driven paths.
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    console = Console()
    live_flags = [
        args.live_open,
        args.live_tick,
        args.live_status,
        args.live_auto_promote,
    ]
    if sum(bool(flag) for flag in live_flags) > 1:
        parser.error(
            "--live-open, --live-tick, --live-status, and --live-auto-promote "
            "are mutually exclusive"
        )

    if args.data_health:
        return _run_data_health_mode(console, args, parser)

    if args.combined_book:
        return _run_combined_book_mode(console, args, parser)

    if args.paper_trade:
        return _run_paper_trade_mode(console, inception=args.inception or None)

    if args.paper_status:
        return _run_paper_status_mode(console)

    if args.live_open:
        return _run_live_open_mode(console, inception=args.inception or None)

    if args.live_tick:
        return _run_live_tick_mode(
            console, book_id=args.book_id or None, as_of=args.as_of or None
        )

    if args.live_status:
        return _run_live_status_mode(console, book_id=args.book_id or None)

    if args.live_auto_promote:
        return _run_live_auto_promote_mode(console)

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

    if args.budget:
        from trading_research_agent.tools.trial_budget import assess_trial_budget, format_budget

        budget = assess_trial_budget(load_history())
        console.print(Panel(format_budget(budget), title="Cumulative Trial Budget"))
        return 0

    if args.macro_regime:
        return _run_macro_regime_mode(console, args, parser)

    if args.sweep:
        return _run_sweep_mode(console, args, parser)

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


def _run_macro_regime_mode(console: Console, args: Any, parser: argparse.ArgumentParser) -> int:
    from trading_research_agent.workflows.macro_regime import format_macro_regime, run_macro_regime

    if not args.start or not args.end:
        parser.error("--macro-regime requires --start and --end")
    if args.lockbox_pct <= 0:
        parser.error("--macro-regime requires --lockbox-pct > 0 (e.g. --lockbox-pct 0.25)")

    try:
        result = run_macro_regime(args.macro_regime, args.start, args.end, args.lockbox_pct)
    except Exception as exc:
        console.print(Panel(f"Macro-regime run failed: {exc}", title="Macro Regime"))
        return 1

    console.print(Panel(format_macro_regime(result), title="Macro Regime — frozen mapping"))
    return 0


def _run_sweep_mode(console: Console, args: Any, parser: argparse.ArgumentParser) -> int:
    from trading_research_agent.schemas.strategy import StrategyFamily
    from trading_research_agent.workflows.parameter_sweep import (
        format_sweep,
        run_single_asset_sweep,
    )

    if not args.sweep_asset or not args.start or not args.end:
        parser.error("--sweep requires --sweep-asset, --start and --end")
    if args.lockbox_pct <= 0:
        parser.error("--sweep requires --lockbox-pct > 0 (e.g. --lockbox-pct 0.25)")

    raw_values = [v.strip() for v in args.sweep_values.split(",") if v.strip()]
    if len(raw_values) < 2:
        parser.error("--sweep requires --sweep-values with at least 2 values")
    try:
        values = [float(v) for v in raw_values]
    except ValueError:
        parser.error("--sweep-values must be numbers, e.g. 20,30,40,55,70,90")

    family = StrategyFamily(args.sweep_family)
    try:
        result = run_single_asset_sweep(
            asset=args.sweep_asset,
            family=family,
            param=args.sweep,
            values=values,
            start=args.start,
            end=args.end,
            lockbox_pct=args.lockbox_pct,
        )
    except Exception as exc:
        console.print(Panel(f"Sweep failed: {exc}", title="Parameter Sweep"))
        return 1

    console.print(Panel(format_sweep(result), title=f"Parameter Sweep — {result['param']}"))
    return 0


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
    if family == PortfolioFamily.VOLATILITY_SCALED_MOMENTUM:
        return (
            "Positive cross-asset trends may persist, but capital should be sized by "
            "recent volatility so high-volatility assets do not dominate the book."
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


def _run_live_open_mode(console: Console, *, inception: str | None) -> int:
    from trading_research_agent.workflows.live_trading import open_live_book
    from trading_research_agent.workflows.robustness_stress import (
        latest_confirmed_portfolio_winner,
    )

    winner = latest_confirmed_portfolio_winner(load_history())
    if winner is None:
        console.print(
            Panel(
                "No lockbox-confirmed portfolio winner in history to open as a "
                "stateful live paper book. Run a confirmed --portfolio first.",
                title="Live Paper Book",
            )
        )
        return 1

    try:
        book = open_live_book(winner, inception=inception)
    except Exception as exc:
        console.print(Panel(f"Could not open live paper book: {exc}", title="Live Paper Book"))
        return 1

    exp = book["expectation"]
    console.print(
        Panel(
            "\n".join(
                [
                    f"Book id:      {book['book_id']}",
                    f"Strategy:     {book['strategy_family']}",
                    f"Universe:     {', '.join(book['params']['assets'])}",
                    f"Inception:    {book['inception_date']}",
                    f"Expectation:  {exp['annualized_return_pct']:.1f}% annualized "
                    f"(backtest max drawdown {exp['backtest_max_drawdown_pct']:.1f}%)",
                    "",
                    "Stateful paper book opened. Run --live-tick after each new "
                    "market bar to persist cash, positions, and the ledger.",
                ]
            ),
            title="Live Paper Book — opened",
        )
    )
    return 0


def _run_live_tick_mode(
    console: Console, *, book_id: str | None, as_of: str | None
) -> int:
    from trading_research_agent.workflows.live_trading import (
        evaluate_live_book,
        list_open_books,
        load_book,
        tick_live_book,
    )

    if book_id:
        book = load_book(book_id)
        if book is None:
            console.print(Panel(f"No live paper book found for id {book_id}.", title="Live Tick"))
            return 1
        if book.get("status") != "open":
            console.print(Panel(f"Book {book_id} is not open.", title="Live Tick"))
            return 1
        books = [book]
    else:
        books = list_open_books()

    if not books:
        console.print(
            Panel("No open live paper books. Open one with --live-open.", title="Live Tick")
        )
        return 0

    had_error = False
    for book in books:
        before = len(book.get("ledger", []))
        try:
            updated = tick_live_book(book, as_of=as_of)
            ev = evaluate_live_book(updated)
        except Exception as exc:
            had_error = True
            console.print(
                Panel(
                    f"Could not tick book {book.get('book_id', '?')}: {exc}",
                    title="Live Tick",
                )
            )
            continue

        added = len(updated.get("ledger", [])) - before
        console.rule(f"Live paper book {updated['book_id']}")
        console.print(
            Panel(
                _format_live_book_evaluation(updated, ev, added_bars=added),
                title="Live Tick — state updated",
            )
        )
    return 1 if had_error else 0


def _run_live_status_mode(console: Console, *, book_id: str | None) -> int:
    from trading_research_agent.workflows.live_trading import (
        evaluate_live_book,
        list_books,
        load_book,
    )

    if book_id:
        book = load_book(book_id)
        if book is None:
            console.print(Panel(f"No live paper book found for id {book_id}.", title="Live Status"))
            return 1
        books = [book]
    else:
        books = list_books()

    if not books:
        console.print(
            Panel("No live paper books. Open one with --live-open.", title="Live Status")
        )
        return 0

    for book in books:
        ev = evaluate_live_book(book)
        console.rule(f"Live paper book {book['book_id']}")
        console.print(
            Panel(
                _format_live_book_evaluation(book, ev),
                title="Live Status",
            )
        )
    return 0


def _run_live_auto_promote_mode(console: Console) -> int:
    from trading_research_agent.workflows.live_trading import auto_promote

    try:
        result = auto_promote(load_history())
    except Exception as exc:
        console.print(Panel(f"Live auto-promotion failed: {exc}", title="Live Auto-Promote"))
        return 1

    action = result.get("action")
    if action == "promoted":
        opened = ", ".join(result.get("opened", []))
        detail = [
            f"Opened new live paper book(s): {opened}",
            f"Winner timestamp: {result.get('winner_ts', '?')}",
            "",
            "Run --live-tick to advance the new book through accrued bars.",
        ]
    elif action == "no_open_books":
        detail = [
            "No open live paper books exist yet.",
            "Open the first one with --live-open; auto-promotion only compares "
            "new confirmed winners against existing open books.",
        ]
    else:
        detail = ["No newer lockbox-confirmed winner found."]

    console.print(Panel("\n".join(detail), title="Live Auto-Promote"))
    return 0


def _format_live_book_evaluation(
    book: dict[str, Any],
    ev: dict[str, Any],
    *,
    added_bars: int | None = None,
) -> str:
    lines = [
        f"Book id:       {book['book_id']}",
        f"Status:        {book.get('status', 'unknown')}",
        f"Strategy:      {book['strategy_family']}",
        f"Universe:      {', '.join(book['params']['assets'])}",
        f"Inception:     {book['inception_date']}",
        f"Last bar:      {book.get('last_bar_date') or 'none'}",
    ]
    if added_bars is not None:
        lines.append(f"Bars added:    {added_bars}")

    if ev["status"] == "no_bars_yet":
        lines.extend(["", ev["detail"]])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            f"NAV:           {ev['nav']:,.2f}",
            f"Cash:          {float(ev['cash']):,.2f}",
            f"Positions:     {_format_live_positions(ev['positions'])}",
            "",
            f"Realized ret:  {ev['realized_return_pct']:7.1f}%",
            f"Annualized:    {ev['realized_annualized_pct']:7.1f}%   "
            f"(backtest expected {ev['expected_annualized_pct']:.1f}%)",
            f"Max DD:        {ev['realized_max_drawdown_pct']:7.1f}%   "
            f"(backtest worst {ev['backtest_max_drawdown_pct']:.1f}%)",
            f"Bars:          {ev['forward_trading_days']}",
            "",
            f"READ: {ev['read']}",
            ev["detail"],
        ]
    )
    return "\n".join(lines)


def _format_live_positions(positions: dict[str, Any]) -> str:
    nonzero = [
        f"{asset}={float(shares):.4f}"
        for asset, shares in positions.items()
        if abs(float(shares)) > 1e-8
    ]
    return ", ".join(nonzero) if nonzero else "all cash"








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














































if __name__ == "__main__":
    raise SystemExit(main())
