"""Rich-console rendering helpers for the trade-research CLI.

Pure presentation: every function takes a Console plus already-computed data and
prints. Extracted from app.py to keep the CLI shell thin.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_research_agent.workflows.campaign_research import CampaignResult
from trading_research_agent.workflows.explore_research import ExploreResult


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


def _print_history_detail_hint(console: Console, slate_id: str | None) -> None:
    if not slate_id:
        return
    console.print(
        Panel(
            f"Run id: {slate_id}\nDetails: trade-research --history-detail {slate_id}",
            title="History",
        )
    )


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
