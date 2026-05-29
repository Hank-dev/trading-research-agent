import json
from pathlib import Path

import pandas as pd

from trading_research_agent.reports import html_dashboard as hd


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_build_report_with_empty_history(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    history.write_text("", encoding="utf-8")
    paper.write_text("", encoding="utf-8")
    out = tmp_path / "dashboard.html"

    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)

    html = Path(path).read_text(encoding="utf-8")
    assert "Trading Research Dashboard" in html
    assert "No research history yet" in html
    assert "No open paper positions" in html


def test_build_report_renders_history_sections(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    paper.write_text("", encoding="utf-8")
    write_jsonl(
        history,
        [
            {
                "timestamp": "2026-05-01T00:00:00Z", "is_lockbox": False,
                "asset": "BTC-USD", "strategy_family": "sma_crossover", "verdict": "reject",
                "failed_checks": ["Benchmark comparison", "Deflated Sharpe ratio (DSR)"],
            },
            {
                "timestamp": "2026-05-02T00:00:00Z", "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]", "strategy_family": "cross_sectional_momentum",
                "verdict": "worth_paper_trading", "failed_checks": [],
            },
        ],
    )
    out = tmp_path / "dashboard.html"

    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)
    html = Path(path).read_text(encoding="utf-8")

    # Embedded charts present, and the confirmed run surfaced.
    assert "data:image/png;base64," in html
    assert "Research funnel" in html
    assert "Verdict distribution" in html
    assert "Recent runs" in html
    assert "cross_sectional_momentum" in html
    assert "Confirmed strategies" in html


def test_build_report_shows_recent_unconfirmed_portfolio_run(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    paper.write_text("", encoding="utf-8")
    write_jsonl(
        history,
        [
            {
                "timestamp": "2026-05-29T12:46:46Z",
                "is_lockbox": False,
                "asset": "PORTFOLIO[GLD,SPY,USO,TLT,BTC-USD]",
                "strategy_family": "cross_sectional_momentum",
                "verdict": "needs_more_testing",
                "failed_checks": ["Benchmark comparison"],
                "metrics": {
                    "total_return_pct": 1655.4023,
                    "buy_and_hold_return_pct": 3305.2412,
                    "sharpe_ratio": 1.408,
                },
            },
            {
                "timestamp": "2026-05-29T12:46:47Z",
                "is_lockbox": True,
                "asset": "PORTFOLIO[GLD,SPY,USO,TLT,BTC-USD]",
                "strategy_family": "cross_sectional_momentum",
                "verdict": "needs_more_testing",
                "failed_checks": ["Portfolio walk-forward stability"],
                "metrics": {
                    "total_return_pct": 90.177,
                    "buy_and_hold_return_pct": 59.499,
                    "sharpe_ratio": 1.747,
                },
            },
        ],
    )

    out = tmp_path / "dashboard.html"
    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)
    html = Path(path).read_text(encoding="utf-8")

    assert "Recent runs" in html
    assert "table-scroll" in html
    assert "PORTFOLIO[GLD,SPY,USO,TLT,BTC-USD]" in html
    assert "needs_more_testing" in html
    assert "lockbox" in html
    assert "90.2%" in html


def test_research_funnel_counts_lockbox_paper_and_robust(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    write_jsonl(
        history,
        [
            {
                "timestamp": "2026-05-01T00:00:00Z",
                "mode": "portfolio",
                "slate_id": "abc123",
                "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]",
                "strategy_family": "cross_sectional_momentum",
                "verdict": "worth_paper_trading",
            },
            {
                "timestamp": "2026-05-01T00:00:01Z",
                "mode": "portfolio",
                "slate_id": "abc123",
                "is_lockbox": True,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]",
                "strategy_family": "cross_sectional_momentum",
                "verdict": "worth_paper_trading",
            },
            {
                "timestamp": "2026-05-01T00:00:02Z",
                "mode": "stress",
                "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]",
                "strategy_family": "cross_sectional_momentum",
                "verdict": "ROBUST",
            },
        ],
    )
    write_jsonl(
        paper,
        [
            {
                "id": "p1",
                "status": "closed",
                "inception_date": "2025-01-01",
                "strategy_family": "cross_sectional_momentum",
                "params": {"assets": ["SPY", "TLT", "DBC", "GLD"]},
            }
        ],
    )

    out = tmp_path / "dashboard.html"
    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)
    html = Path(path).read_text(encoding="utf-8")

    assert "Research funnel" in html
    assert "Lockbox-confirmed" in html
    assert "Paper-traded" in html
    assert "Robust after stress" in html
    assert "abc123" in html
    assert "stress" in html


def test_paper_section_handles_no_data_position(tmp_path: Path, monkeypatch) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    history.write_text("", encoding="utf-8")
    write_jsonl(
        paper,
        [
            {
                "id": "p1", "status": "open", "inception_date": "2030-01-01",
                "strategy_family": "cross_sectional_momentum",
                "params": {"assets": ["SPY", "TLT"], "lookback_days": 126, "top_k": 1, "rebalance_days": 21},
                "expectation": {"annualized_return_pct": 8.0, "backtest_max_drawdown_pct": -15.0},
            }
        ],
    )
    # Avoid any network: force evaluate to report no data and curve to be None.
    from trading_research_agent.workflows import paper_trading as pt

    monkeypatch.setattr(
        pt,
        "evaluate_paper_position",
        lambda p, as_of=None: {
            "id": p["id"],
            "status": "no_data_yet",
            "detail": "<urlopen error [Errno -3] Temporary failure>",
        },
    )
    monkeypatch.setattr(pt, "forward_equity_series", lambda p, as_of=None: None)

    out = tmp_path / "dashboard.html"
    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)
    html = Path(path).read_text(encoding="utf-8")

    assert "Forward paper trading" in html
    assert "&lt;urlopen error [Errno -3] Temporary failure&gt;" in html
    assert "<urlopen error" not in html


def test_paper_section_renders_pnl_curve_when_evaluated(tmp_path: Path, monkeypatch) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    history.write_text("", encoding="utf-8")
    write_jsonl(
        paper,
        [
            {
                "id": "p1",
                "status": "open",
                "inception_date": "2025-01-01",
                "strategy_family": "cross_sectional_momentum",
                "params": {
                    "assets": ["SPY", "TLT", "DBC", "GLD"],
                    "lookback_days": 126,
                    "top_k": 2,
                    "rebalance_days": 21,
                },
                "expectation": {
                    "annualized_return_pct": 8.0,
                    "backtest_max_drawdown_pct": -15.0,
                },
            }
        ],
    )

    from trading_research_agent.workflows import paper_trading as pt

    monkeypatch.setattr(
        pt,
        "evaluate_paper_position",
        lambda p, as_of=None: {
            "id": p["id"],
            "status": "evaluated",
            "inception_date": p["inception_date"],
            "as_of": "2025-04-01",
            "forward_trading_days": 64,
            "realized_return_pct": 6.0,
            "realized_annualized_pct": 24.0,
            "realized_max_drawdown_pct": -4.0,
            "expected_annualized_pct": 8.0,
            "backtest_max_drawdown_pct": -15.0,
            "read": "TRACKING",
            "detail": "forward run is tracking",
        },
    )
    monkeypatch.setattr(
        pt,
        "forward_equity_series",
        lambda p, as_of=None: pd.Series(
            [10000.0, 10200.0, 10600.0],
            index=pd.to_datetime(["2025-01-01", "2025-02-01", "2025-04-01"]),
        ),
    )

    out = tmp_path / "dashboard.html"
    path = hd.build_html_report(output_path=out, history_path=history, paper_path=paper)
    html = Path(path).read_text(encoding="utf-8")

    assert "PnL curve" in html
    assert "data:image/png;base64," in html
    assert "Realized return: <b>6.0%</b>" in html
