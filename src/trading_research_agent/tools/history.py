"""Append-only research history.

One JSONL row per backtested hypothesis — including each candidate inside an
`--explore` slate and the held-out lockbox re-test. Lets you see across sessions
what has been tried and which gates have been hardest to clear, without an LLM
in the loop.

The file lives at `outputs/history.jsonl` by default, or under
`TRADING_RESEARCH_OUTPUT_DIR` when configured for deployment.
"""

from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from trading_research_agent.config import get_output_path
from trading_research_agent.schemas.strategy import StrategySpec


HISTORY_FILENAME = "history.jsonl"
HISTORY_PATH = get_output_path(HISTORY_FILENAME)


def default_history_path() -> Path:
    return get_output_path(HISTORY_FILENAME)


def append_run_record(record: dict[str, Any], path: Path | None = None) -> None:
    path = path or default_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_history(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or default_history_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def record_from_state(
    state: dict[str, Any],
    *,
    mode: str,
    user_request: str,
    slate_id: str | None = None,
    is_lockbox: bool = False,
) -> dict[str, Any] | None:
    """Build a JSON-serializable record from a research-graph state. Returns
    None if the state has nothing meaningful to log (e.g. parsing failed)."""
    spec: StrategySpec | None = state.get("strategy_spec")
    result = state.get("backtest_result")
    report = state.get("report")

    if spec is None and result is None and report is None:
        return None

    metrics_dict: dict[str, Any] | None = None
    failed_checks: list[str] = []
    if result is not None:
        m = result.metrics
        metrics_dict = {
            "total_return_pct": m.total_return_pct,
            "buy_and_hold_return_pct": m.buy_and_hold_return_pct,
            "sharpe_ratio": m.sharpe_ratio,
            "max_drawdown_pct": m.max_drawdown_pct,
            "num_trades": m.num_trades,
            "beats_benchmark": m.beats_benchmark,
        }
        failed_checks = [c.test_name for c in result.robustness_results if not c.passed]

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "user_request": user_request,
        "is_lockbox": is_lockbox,
    }
    if slate_id is not None:
        record["slate_id"] = slate_id
    if spec is not None:
        record.update(_spec_record(spec))
    if metrics_dict is not None:
        record["metrics"] = metrics_dict
    if failed_checks:
        record["failed_checks"] = failed_checks
    if report is not None:
        record["verdict"] = report.verdict

    return record


def _spec_record(spec: Any) -> dict[str, Any]:
    """Build the spec portion of a record, handling both single-asset
    StrategySpec and multi-asset PortfolioSpec."""
    if hasattr(spec, "portfolio_family"):
        return {
            "asset": "PORTFOLIO[" + ",".join(spec.assets) + "]",
            "strategy_family": spec.portfolio_family.value,
            "data_source": spec.data_source.value if spec.data_source else "auto",
            "start_date": spec.start_date,
            "end_date": spec.end_date,
            "params": {
                "assets": list(spec.assets),
                "lookback_days": spec.lookback_days,
                "top_k": spec.top_k,
                "rebalance_days": spec.rebalance_days,
            },
        }
    return {
        "asset": spec.asset,
        "strategy_family": spec.strategy_family.value,
        "data_source": spec.data_source.value,
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "params": _spec_params(spec),
    }


def _spec_params(spec: StrategySpec) -> dict[str, Any]:
    """Strategy-family-specific parameter dict for the record."""
    fields = (
        "fast_window",
        "slow_window",
        "entry_window",
        "exit_window",
        "rsi_window",
        "oversold_threshold",
        "exit_threshold",
    )
    return {name: getattr(spec, name) for name in fields if getattr(spec, name) is not None}


def summarize_history(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-run summary. Aggregates only over non-lockbox records (lockbox runs
    are verification, not search trials), but reports the lockbox count separately."""
    trials = [
        r
        for r in records
        if not r.get("is_lockbox", False) and r.get("mode") != "stress"
    ]
    lockbox_runs = [r for r in records if r.get("is_lockbox", False)]
    stress_runs = [r for r in records if r.get("mode") == "stress"]

    by_asset: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_verdict: Counter[str] = Counter()
    failed_checks: Counter[str] = Counter()
    asset_family_pairs: Counter[tuple[str, str]] = Counter()
    passed_runs: list[dict[str, Any]] = []

    for r in trials:
        asset = r.get("asset", "unknown")
        family = r.get("strategy_family", "unknown")
        verdict = r.get("verdict", "unknown")
        by_asset[asset] += 1
        by_family[family] += 1
        by_verdict[verdict] += 1
        asset_family_pairs[(asset, family)] += 1
        for check in r.get("failed_checks", []):
            failed_checks[check] += 1
        if verdict == "worth_paper_trading":
            passed_runs.append(r)

    timestamps = [r["timestamp"] for r in records if "timestamp" in r]
    return {
        "total_trials": len(trials),
        "lockbox_runs": len(lockbox_runs),
        "stress_runs": len(stress_runs),
        "robust_stress_runs": sum(1 for r in stress_runs if r.get("verdict") == "ROBUST"),
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "by_asset": dict(by_asset.most_common()),
        "by_family": dict(by_family.most_common()),
        "by_verdict": dict(by_verdict.most_common()),
        "failed_checks": dict(failed_checks.most_common()),
        "asset_family_pairs": {
            f"{asset} / {family}": count
            for (asset, family), count in asset_family_pairs.most_common()
        },
        "passed_runs": passed_runs,
    }


def format_summary(summary: dict[str, Any]) -> str:
    if summary["total_trials"] == 0:
        return f"No history yet. Run some backtests to populate {default_history_path()}."

    lines: list[str] = []
    span = ""
    if summary["earliest_timestamp"] and summary["latest_timestamp"]:
        span = f" between {summary['earliest_timestamp']} and {summary['latest_timestamp']}"
    lines.append(
        f"History: {summary['total_trials']} trial(s){span} "
        f"(+ {summary['lockbox_runs']} lockbox verification(s), "
        f"+ {summary.get('stress_runs', 0)} stress test(s))"
    )
    lines.append("")

    if summary["by_verdict"]:
        lines.append("By verdict:")
        for verdict, count in summary["by_verdict"].items():
            lines.append(f"  {count:>4}x  {verdict}")
        lines.append("")

    if summary["by_asset"]:
        lines.append("By asset:")
        for asset, count in summary["by_asset"].items():
            lines.append(f"  {count:>4}x  {asset}")
        lines.append("")

    if summary["by_family"]:
        lines.append("By strategy family:")
        for family, count in summary["by_family"].items():
            lines.append(f"  {count:>4}x  {family}")
        lines.append("")

    if summary["asset_family_pairs"]:
        lines.append("By asset / family pair:")
        for pair, count in summary["asset_family_pairs"].items():
            lines.append(f"  {count:>4}x  {pair}")
        lines.append("")

    if summary["failed_checks"]:
        lines.append("Most-failed robustness checks across all trials:")
        for check, count in summary["failed_checks"].items():
            lines.append(f"  {count:>4}x  {check}")
        lines.append("")

    if summary["passed_runs"]:
        lines.append(f"Runs that reached worth_paper_trading ({len(summary['passed_runs'])}):")
        for r in summary["passed_runs"]:
            asset = r.get("asset", "?")
            family = r.get("strategy_family", "?")
            ts = r.get("timestamp", "?")
            lines.append(f"  {ts}  {asset} {family}")
        lines.append("")
    else:
        lines.append("No runs have reached worth_paper_trading yet.")
        lines.append("")

    lines.append("Cross-run multiple-testing note:")
    lines.append(
        "  Each --explore slate's Deflated Sharpe is computed against that slate only."
    )
    lines.append(
        f"  Your true cumulative trial count is {summary['total_trials']} — the more you"
    )
    lines.append(
        "  search across sessions, the higher the bar a strategy must clear to be real."
    )

    return "\n".join(lines)
