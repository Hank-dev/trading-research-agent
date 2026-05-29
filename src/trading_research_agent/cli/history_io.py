"""History-logging and report-saving helpers for the trade-research CLI.

Side-effecting persistence, best-effort by design: a logging or dashboard
failure never sinks a research run. Extracted from app.py.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel

from trading_research_agent.reports.markdown_report import save_markdown_report
from trading_research_agent.schemas.portfolio import PortfolioSpec
from trading_research_agent.tools.history import append_run_record, record_from_state
from trading_research_agent.workflows.campaign_research import CampaignResult
from trading_research_agent.workflows.explore_research import ExploreResult


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
