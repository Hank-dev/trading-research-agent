"""Readable detail reports reconstructed from append-only research history."""

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from trading_research_agent.tools.history import HISTORY_PATH, load_history
from trading_research_agent.workflows.paper_trading import PAPER_PATH, load_paper_positions


def build_history_detail(
    identifier: str,
    *,
    history_path: Path = HISTORY_PATH,
    paper_path: Path = PAPER_PATH,
) -> dict[str, Any]:
    records = load_history(path=history_path)
    if not records:
        return {"status": "no_history", "identifier": identifier, "records": []}

    groups = _groups(records)
    matches = _matching_groups(identifier, groups)
    if not matches:
        return {"status": "not_found", "identifier": identifier, "records": []}

    # Show the newest matching run. The report says how many groups matched so
    # broad searches are still honest.
    selected_id, selected = matches[0]

    positions = [
        p
        for p in load_paper_positions(path=paper_path)
        if _paper_matches_records(p, selected)
    ]

    return {
        "status": "ok",
        "identifier": identifier,
        "matched_groups": len(matches),
        "group_id": selected_id,
        "records": sorted(selected, key=lambda r: (bool(r.get("is_lockbox")), r.get("timestamp", ""))),
        "paper_positions": positions,
    }


def render_history_detail_markdown(detail: dict[str, Any]) -> str:
    status = detail.get("status")
    identifier = detail.get("identifier", "")
    if status == "no_history":
        return "No history yet. Run some backtests to populate outputs/history.jsonl."
    if status == "not_found":
        return f"No history record matched `{identifier}`."
    if status != "ok":
        return f"Could not build history detail for `{identifier}`."

    records = detail["records"]
    primary = _primary_record(records)
    family = primary.get("strategy_family", "?")
    assets = _asset_label(primary)

    lines: list[str] = [
        f"# Research Detail: {family} - {assets}",
        "",
        "## Identity",
        f"- Query: `{identifier}`",
        f"- Run id: `{detail.get('group_id', '-')}`",
        f"- Matching groups found: {detail.get('matched_groups', 1)}",
        f"- Mode: {primary.get('mode', '?')}",
        f"- User request: {primary.get('user_request', '-')}",
        "",
        "## Strategy",
        f"- Family: `{family}`",
        f"- Assets: {assets}",
        f"- Parameters: `{_params_text(primary.get('params') or {})}`",
        "",
        "## Evidence",
        "| Phase | Window | Verdict | Return | Benchmark | Sharpe | Max DD | Trades | Failed checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for record in records:
        metrics = record.get("metrics") or {}
        phase = "lockbox" if record.get("is_lockbox") else record.get("mode", "trial")
        failed = ", ".join(record.get("failed_checks", [])) or "-"
        lines.append(
            "| "
            f"{phase} | "
            f"{record.get('start_date', '?')} to {record.get('end_date', '?')} | "
            f"{record.get('verdict', '?')} | "
            f"{_fmt_pct(metrics.get('total_return_pct'))} | "
            f"{_fmt_pct(metrics.get('buy_and_hold_return_pct'))} | "
            f"{_fmt_num(metrics.get('sharpe_ratio'))} | "
            f"{_fmt_pct(metrics.get('max_drawdown_pct'))} | "
            f"{metrics.get('num_trades', '-')} | "
            f"{failed} |"
        )

    lines.extend(["", "## Read"])
    lockbox = next((r for r in records if r.get("is_lockbox")), None)
    if lockbox is None:
        lines.append("- No lockbox re-test is recorded for this run.")
    elif lockbox.get("verdict") == "worth_paper_trading":
        lines.append("- The strategy confirmed on the held-out lockbox.")
        lines.append("- Next robustness gate: `trade-research --stress`.")
    else:
        lines.append(
            f"- The held-out lockbox verdict was `{lockbox.get('verdict', '?')}`; "
            "do not treat the in-slate result as confirmed."
        )

    positions = detail.get("paper_positions") or []
    lines.extend(["", "## Paper Trading"])
    if positions:
        for pos in positions:
            lines.append(
                "- "
                f"`{pos.get('id', '?')}` status={pos.get('status', '?')} "
                f"inception={pos.get('inception_date', '?')}"
            )
    else:
        lines.append("- No matching paper position is recorded.")

    return "\n".join(lines)


def save_history_detail_report(
    markdown: str,
    identifier: str,
    *,
    output_dir: Path = Path("outputs"),
) -> str:
    output_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", identifier).strip("_").lower() or "history_detail"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_{safe}_history_detail.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def _groups(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        group_id = str(record.get("slate_id") or f"record-{index + 1}")
        key = (group_id, _strategy_key(record))
        by_key.setdefault(key, []).append(record)
    return [(group_id, recs) for (group_id, _strategy), recs in by_key.items()]


def _strategy_key(record: dict[str, Any]) -> str:
    params = record.get("params") or {}
    parts = [
        str(record.get("strategy_family", "")),
        ",".join(_assets_from_record(record)),
        str(params.get("lookback_days", "")),
        str(params.get("top_k", "")),
        str(params.get("rebalance_days", "")),
    ]
    return "|".join(parts)


def _matching_groups(
    identifier: str, groups: list[tuple[str, list[dict[str, Any]]]]
) -> list[tuple[str, list[dict[str, Any]]]]:
    query = identifier.strip().lower()
    if not query:
        return []

    exact = [(group_id, recs) for group_id, recs in groups if group_id.lower() == query]
    if exact:
        return exact

    query_tokens = _tokens(query)
    known_assets = {
        asset.lower()
        for _group_id, recs in groups
        for record in recs
        for asset in _assets_from_record(record)
    }
    query_assets = {token for token in query_tokens if token in known_assets}

    scored: list[tuple[int, str, list[dict[str, Any]]]] = []
    for group_id, recs in groups:
        haystack = " ".join(_record_text(r) for r in recs).lower()
        score = _match_score(query, query_tokens, query_assets, haystack, recs)
        if score > 0:
            scored.append((score, group_id, recs))

    scored.sort(
        key=lambda item: (item[0], max(r.get("timestamp", "") for r in item[2])),
        reverse=True,
    )
    return [(group_id, recs) for _score, group_id, recs in scored]


def _match_score(
    query: str,
    query_tokens: list[str],
    query_assets: set[str],
    haystack: str,
    records: list[dict[str, Any]],
) -> int:
    score = 0
    group_assets = {
        asset.lower() for record in records for asset in _assets_from_record(record)
    }

    if query_assets:
        if not query_assets.issubset(group_assets):
            return 0
        if query_assets == group_assets:
            score += 1000
        else:
            score += max(100, 500 - (len(group_assets) - len(query_assets)))

    if query in haystack:
        score += 20
    if query_tokens and all(t in haystack for t in query_tokens):
        score += 10

    families = {str(r.get("strategy_family", "")).lower() for r in records}
    if any(family and family in query_tokens for family in families):
        score += 50

    return score


def _record_text(record: dict[str, Any]) -> str:
    assets = _assets_from_record(record)
    return " ".join(
        [
            str(record.get("slate_id", "")),
            str(record.get("asset", "")),
            str(record.get("strategy_family", "")),
            " ".join(str(a) for a in assets),
        ]
    )


def _assets_from_record(record: dict[str, Any]) -> list[str]:
    params = record.get("params") or {}
    assets = params.get("assets")
    if isinstance(assets, list):
        return [str(a) for a in assets]

    asset = str(record.get("asset", ""))
    if asset.startswith("PORTFOLIO[") and asset.endswith("]"):
        inner = asset[len("PORTFOLIO[") : -1]
        return [part.strip() for part in inner.split(",") if part.strip()]
    return [asset] if asset else []


def _tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_.-]+", value.lower()) if len(t) > 1]


def _primary_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next((r for r in records if not r.get("is_lockbox")), records[0])


def _asset_label(record: dict[str, Any]) -> str:
    params = record.get("params") or {}
    assets = params.get("assets")
    if assets:
        return ", ".join(str(a) for a in assets)
    return str(record.get("asset", "?"))


def _params_text(params: dict[str, Any]) -> str:
    items = [
        (key, value)
        for key, value in params.items()
        if key != "assets" and value is not None
    ]
    return ", ".join(f"{key}={value}" for key, value in items) or "-"


def _paper_matches_records(position: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    params = position.get("params") or {}
    paper_assets = [str(a).upper() for a in params.get("assets", [])]
    paper_family = position.get("strategy_family")
    if not paper_assets or not paper_family:
        return False

    for record in records:
        record_params = record.get("params") or {}
        record_assets = [str(a).upper() for a in record_params.get("assets", [])]
        if record_assets == paper_assets and record.get("strategy_family") == paper_family:
            return True
    return False


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_num(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"
