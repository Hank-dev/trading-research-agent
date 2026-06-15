"""Static, self-contained HTML research dashboard.

Read-only by design: it visualizes what the research engine has already produced
(cross-session history, verdicts, and forward paper-trade curves) but offers no
controls to tweak-and-re-run. Strategy creation stays in the disciplined CLI,
where the pre-registration friction lives — a point-and-click "nudge the
parameter, watch the Sharpe" panel is exactly the overfitting ergonomic this
project avoids.

Charts are rendered with matplotlib and inlined as base64 PNGs, so the output is
a single file that opens in any browser with no server and no internet.
"""

import base64
from datetime import UTC, datetime
from html import escape
import io
from pathlib import Path
from typing import Any

from trading_research_agent.config import get_output_path
from trading_research_agent.tools.history import load_history, summarize_history

# Reuse the matplotlib Agg setup from the plotting module (sets MPLCONFIGDIR + backend).
from trading_research_agent.tools.plotting import plt

_VERDICT_COLORS = {
    "worth_paper_trading": "#1a9850",
    "needs_more_testing": "#f0a830",
    "reject": "#d73027",
    "error": "#777777",
}


def build_html_report(
    output_path: str | Path | None = None,
    history_path: Path | None = None,
    paper_path: Path | None = None,
) -> str:
    records = load_history(**({"path": history_path} if history_path else {}))
    summary = summarize_history(records)

    sections = [
        _summary_section(summary),
        _research_funnel_section(records, paper_path),
        _recent_runs_section(records),
        _verdict_section(summary),
        _breakdown_section(summary),
        _failed_checks_section(summary),
        _passed_runs_section(summary),
        _paper_section(paper_path),
    ]

    html = _PAGE_TEMPLATE.format(
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        body="\n".join(sections),
    )
    path = Path(output_path) if output_path is not None else get_output_path("dashboard.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)


def _summary_section(summary: dict[str, Any]) -> str:
    if summary["total_trials"] == 0:
        return _card("Overview", "<p>No research history yet. Run some backtests to populate the dashboard.</p>")
    span = ""
    if summary["earliest_timestamp"] and summary["latest_timestamp"]:
        span = f"{summary['earliest_timestamp']} &rarr; {summary['latest_timestamp']}"
    confirmed = len(summary["passed_runs"])
    tiles = [
        _tile("Total trials", summary["total_trials"]),
        _tile("Lockbox verifications", summary["lockbox_runs"]),
        _tile("Reached worth_paper_trading", confirmed),
        _tile("Assets explored", len(summary["by_asset"])),
    ]
    return _card("Overview", f'<div class="tiles">{"".join(tiles)}</div><p class="muted">{span}</p>')


def _research_funnel_section(records: list[dict[str, Any]], paper_path: Path | None) -> str:
    from trading_research_agent.workflows.paper_trading import load_paper_positions

    trials = [
        r
        for r in records
        if not r.get("is_lockbox", False) and r.get("mode") != "stress"
    ]
    lockboxes = [r for r in records if r.get("is_lockbox")]
    confirmed = [r for r in lockboxes if r.get("verdict") == "worth_paper_trading"]
    stress_robust = [
        r for r in records if r.get("mode") == "stress" and r.get("verdict") == "ROBUST"
    ]
    try:
        paper_positions = load_paper_positions(**({"path": paper_path} if paper_path else {}))
    except Exception:
        paper_positions = []

    tiles = [
        _tile("Tried", len(trials)),
        _tile("Lockbox-tested", len(lockboxes)),
        _tile("Lockbox-confirmed", len(confirmed)),
        _tile("Paper-traded", len(paper_positions)),
        _tile("Robust after stress", len(stress_robust)),
    ]
    return _card(
        "Research funnel",
        f'<div class="tiles">{"".join(tiles)}</div>'
        '<p class="muted">Counts are sequential gates, not independent proof. '
        'A strategy is strongest only after lockbox confirmation, stress survival, '
        'and forward paper evidence.</p>',
    )


def _verdict_section(summary: dict[str, Any]) -> str:
    counts = summary.get("by_verdict", {})
    if not counts:
        return ""
    order = ["worth_paper_trading", "needs_more_testing", "reject", "error"]
    labels = [v for v in order if v in counts] + [v for v in counts if v not in order]
    values = [counts[v] for v in labels]
    colors = [_VERDICT_COLORS.get(v, "#888") for v in labels]
    img = _bar_chart(labels, values, "Verdicts across all trials", colors=colors)
    return _card("Verdict distribution", _img(img))


def _recent_runs_section(records: list[dict[str, Any]], limit: int = 20) -> str:
    if not records:
        return ""

    recent = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)[:limit]
    rows = []
    for r in recent:
        metrics = r.get("metrics") or {}
        phase = (
            "stress"
            if r.get("mode") == "stress"
            else ("lockbox" if r.get("is_lockbox") else "trial")
        )
        failed = ", ".join(r.get("failed_checks", [])) or "-"
        rows.append(
            "<tr>"
            f"<td>{escape(str(r.get('timestamp', '?')))}</td>"
            f"<td>{escape(str(r.get('slate_id', '-')))}</td>"
            f"<td>{escape(phase)}</td>"
            f"<td>{escape(str(r.get('asset', '?')))}</td>"
            f"<td>{escape(str(r.get('strategy_family', '?')))}</td>"
            f"<td>{escape(str(r.get('verdict', '?')))}</td>"
            f"<td>{_fmt_pct(metrics.get('total_return_pct'))}</td>"
            f"<td>{_fmt_pct(metrics.get('buy_and_hold_return_pct'))}</td>"
            f"<td>{_fmt_num(metrics.get('sharpe_ratio'))}</td>"
            f"<td>{escape(failed)}</td>"
            "</tr>"
        )

    table = (
        '<div class="table-scroll"><table class="recent-table"><thead><tr>'
        "<th>When</th><th>Run id</th><th>Phase</th><th>Asset</th><th>Family</th><th>Verdict</th>"
        "<th>Return</th><th>Benchmark</th><th>Sharpe</th><th>Failed checks</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return _card("Recent runs", table)


def _breakdown_section(summary: dict[str, Any]) -> str:
    asset_tbl = _counter_table("Asset", summary.get("by_asset", {}))
    family_tbl = _counter_table("Strategy family", summary.get("by_family", {}))
    if not asset_tbl and not family_tbl:
        return ""
    return _card("Breakdown", f'<div class="two-col"><div>{asset_tbl}</div><div>{family_tbl}</div></div>')


def _failed_checks_section(summary: dict[str, Any]) -> str:
    checks = summary.get("failed_checks", {})
    if not checks:
        return ""
    labels = list(checks.keys())
    values = list(checks.values())
    img = _bar_chart(values, labels, "Most-failed robustness checks", horizontal=True)
    return _card("Where strategies fail", _img(img))


def _passed_runs_section(summary: dict[str, Any]) -> str:
    passed = summary.get("passed_runs", [])
    if not passed:
        return _card(
            "Confirmed strategies",
            '<p class="muted">No run has reached worth_paper_trading yet.</p>',
        )
    rows = "".join(
        f"<tr><td>{r.get('timestamp', '?')}</td><td>{r.get('asset', '?')}</td>"
        f"<td>{r.get('strategy_family', '?')}</td></tr>"
        for r in passed
    )
    table = f"<table><thead><tr><th>When</th><th>Asset</th><th>Family</th></tr></thead><tbody>{rows}</tbody></table>"
    return _card("Confirmed strategies (in-slate)", table)


def _paper_section(paper_path: Path | None) -> str:
    from trading_research_agent.workflows.paper_trading import (
        evaluate_paper_position,
        forward_equity_series,
        load_paper_positions,
    )

    positions = load_paper_positions(**({"path": paper_path} if paper_path else {}))
    open_positions = [p for p in positions if p.get("status") == "open"]
    if not open_positions:
        return _card(
            "Forward paper trading",
            '<p class="muted">No open paper positions. Open one with <code>--paper-trade</code>.</p>',
        )

    blocks: list[str] = []
    for pos in open_positions:
        ev = evaluate_paper_position(pos)
        assets = ", ".join(escape(str(a)) for a in pos["params"]["assets"])
        header = (
            f"<h3>{escape(str(pos['strategy_family']))} &middot; {assets}</h3>"
            f"<p class='muted'>Position {escape(str(pos['id']))} &middot; "
            f"inception {escape(str(pos['inception_date']))}</p>"
        )
        if ev["status"] != "evaluated":
            detail = escape(str(ev.get("detail", ev["status"])))
            blocks.append(header + f"<p>{detail}</p>")
            continue

        read = str(ev["read"])
        read_class = _css_token(read)
        detail = escape(str(ev["detail"]))

        stats = (
            f"<ul>"
            f"<li>Forward window: {ev['forward_trading_days']} trading days "
            f"(to {ev['as_of']})</li>"
            f"<li>Realized return: <b>{ev['realized_return_pct']:.1f}%</b> "
            f"({ev['realized_annualized_pct']:.1f}% annualized)</li>"
            f"<li>Backtest expectation: {ev['expected_annualized_pct']:.1f}% annualized</li>"
            f"<li>Realized max drawdown: {ev['realized_max_drawdown_pct']:.1f}% "
            f"(backtest worst {ev['backtest_max_drawdown_pct']:.1f}%)</li>"
            f"<li>Read: <b class='read-{read_class}'>{escape(read)}</b></li>"
            f"</ul><p class='muted'>{detail}</p>"
        )

        curve = forward_equity_series(pos)
        chart_html = '<p class="muted">PnL curve unavailable because forward price data could not be loaded.</p>'
        if curve is not None:
            normalized = curve / curve.iloc[0] * 100.0
            img = _line_chart(
                normalized, "Forward PnL since inception (indexed to 100)"
            )
            chart_html = f"<h4>PnL curve</h4>{_img(img)}"

        blocks.append(header + f'<div class="two-col"><div>{stats}</div><div>{chart_html}</div></div>')

    return _card("Forward paper trading (out-of-sample)", "\n".join(blocks))


# ---- chart + html helpers ----


def _bar_chart(
    labels: list, values: list, title: str, *, colors=None, horizontal: bool = False
) -> str:
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(labels) + 1.5)))
    if horizontal:
        ax.barh(range(len(labels)), values, color=colors or "#3b6ea5")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.bar(range(len(labels)), values, color=colors or "#3b6ea5")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _line_chart(series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(ax=ax, color="#3b6ea5")
    ax.axhline(100.0, color="#aaa", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img(b64: str) -> str:
    return f'<img alt="chart" src="data:image/png;base64,{b64}" />'


def _counter_table(label: str, counter: dict) -> str:
    if not counter:
        return ""
    rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{v}</td></tr>" for k, v in counter.items()
    )
    return (
        f"<table><thead><tr><th>{escape(label)}</th><th>Trials</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _tile(label: str, value: Any) -> str:
    return (
        f'<div class="tile"><div class="tile-value">{escape(str(value))}</div>'
        f'<div class="tile-label">{escape(label)}</div></div>'
    )


def _card(title: str, body: str) -> str:
    return f'<section class="card"><h2>{escape(title)}</h2>{body}</section>'


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


def _css_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in value)


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Trading Research Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f4f6f8; color: #1c2733; }}
  header {{ background: #1c2733; color: #fff; padding: 20px 28px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .muted {{ color: #9fb0c0; font-size: 13px; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card h2 {{ margin-top: 0; font-size: 16px; border-bottom: 1px solid #eef1f4; padding-bottom: 8px; }}
  .tiles {{ display: flex; gap: 18px; flex-wrap: wrap; }}
  .tile {{ background: #f0f4f8; border-radius: 8px; padding: 14px 18px; min-width: 120px; }}
  .tile-value {{ font-size: 26px; font-weight: 700; }}
  .tile-label {{ font-size: 12px; color: #5a6b7b; }}
  .two-col {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .two-col > div {{ flex: 1; min-width: 280px; }}
  .table-scroll {{ overflow-x: auto; max-width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  .recent-table {{ min-width: 1160px; }}
  .recent-table td {{ vertical-align: top; overflow-wrap: anywhere; }}
  .recent-table th:nth-child(1), .recent-table td:nth-child(1) {{ width: 150px; }}
  .recent-table th:nth-child(4), .recent-table td:nth-child(4) {{ width: 210px; }}
  .recent-table th:nth-child(10), .recent-table td:nth-child(10) {{ width: 250px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eef1f4; }}
  img {{ max-width: 100%; height: auto; }}
  .muted {{ color: #7a8a99; font-size: 13px; }}
  code {{ background: #eef1f4; padding: 1px 5px; border-radius: 4px; }}
  .read-TRACKING {{ color: #1a9850; }}
  .read-TOO_EARLY {{ color: #f0a830; }}
  .read-DIVERGING, .read-DRAWDOWN_BREACH {{ color: #d73027; }}
</style>
</head>
<body>
<header>
  <h1>Trading Research Dashboard</h1>
  <div class="muted">Read-only monitoring view &middot; generated {generated}</div>
</header>
<main>
{body}
</main>
</body>
</html>
"""
