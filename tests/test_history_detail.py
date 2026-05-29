import json
from pathlib import Path

from trading_research_agent.workflows import history_detail as hd


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_history_detail_matches_universe_and_renders_lockbox(tmp_path: Path) -> None:
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
                "start_date": "2010-01-01",
                "end_date": "2021-12-31",
                "verdict": "worth_paper_trading",
                "user_request": "test it",
                "params": {
                    "assets": ["SPY", "TLT", "DBC", "GLD"],
                    "lookback_days": 126,
                    "top_k": 2,
                    "rebalance_days": 21,
                },
                "metrics": {
                    "total_return_pct": 182.78,
                    "buy_and_hold_return_pct": 149.10,
                    "sharpe_ratio": 1.01,
                    "max_drawdown_pct": -16.37,
                    "num_trades": 167,
                },
            },
            {
                "timestamp": "2026-05-01T00:00:01Z",
                "mode": "portfolio",
                "slate_id": "abc123",
                "is_lockbox": True,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]",
                "strategy_family": "cross_sectional_momentum",
                "start_date": "2022-01-01",
                "end_date": "2025-01-01",
                "verdict": "worth_paper_trading",
                "params": {
                    "assets": ["SPY", "TLT", "DBC", "GLD"],
                    "lookback_days": 126,
                    "top_k": 2,
                    "rebalance_days": 21,
                },
                "metrics": {
                    "total_return_pct": 28.46,
                    "buy_and_hold_return_pct": 13.17,
                    "sharpe_ratio": 0.98,
                    "max_drawdown_pct": -11.09,
                    "num_trades": 39,
                },
            },
        ],
    )
    write_jsonl(
        paper,
        [
            {
                "id": "p1",
                "status": "open",
                "inception_date": "2025-01-01",
                "strategy_family": "cross_sectional_momentum",
                "params": {"assets": ["SPY", "TLT", "DBC", "GLD"]},
            }
        ],
    )

    detail = hd.build_history_detail("cross_sectional_momentum - SPY, TLT, DBC, GLD", history_path=history, paper_path=paper)
    markdown = hd.render_history_detail_markdown(detail)

    assert detail["status"] == "ok"
    assert detail["group_id"] == "abc123"
    assert "Run id: `abc123`" in markdown
    assert "2010-01-01 to 2021-12-31" in markdown
    assert "2022-01-01 to 2025-01-01" in markdown
    assert "28.5%" in markdown
    assert "trade-research --stress" in markdown
    assert "`p1` status=open" in markdown


def test_history_detail_reports_not_found(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    write_jsonl(
        history,
        [
            {
                "timestamp": "2026-05-01T00:00:00Z",
                "mode": "portfolio",
                "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,TLT]",
                "strategy_family": "dual_momentum",
                "params": {"assets": ["SPY", "TLT"]},
            }
        ],
    )
    paper.write_text("", encoding="utf-8")

    detail = hd.build_history_detail("GLD,DBC", history_path=history, paper_path=paper)

    assert detail["status"] == "not_found"
    assert "No history record matched" in hd.render_history_detail_markdown(detail)


def test_history_detail_prefers_exact_asset_set_over_newer_superset(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    paper = tmp_path / "paper.jsonl"
    paper.write_text("", encoding="utf-8")
    write_jsonl(
        history,
        [
            {
                "timestamp": "2026-05-01T00:00:00Z",
                "mode": "portfolio",
                "slate_id": "exact",
                "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,TLT,DBC,GLD]",
                "strategy_family": "cross_sectional_momentum",
                "params": {"assets": ["SPY", "TLT", "DBC", "GLD"]},
            },
            {
                "timestamp": "2026-05-02T00:00:00Z",
                "mode": "portfolio",
                "slate_id": "superset",
                "is_lockbox": False,
                "asset": "PORTFOLIO[SPY,QQQ,EFA,EEM,TLT,IEF,GLD,DBC,USO]",
                "strategy_family": "time_series_momentum",
                "params": {
                    "assets": [
                        "SPY",
                        "QQQ",
                        "EFA",
                        "EEM",
                        "TLT",
                        "IEF",
                        "GLD",
                        "DBC",
                        "USO",
                    ]
                },
            },
        ],
    )

    detail = hd.build_history_detail(
        "SPY, TLT, DBC, GLD", history_path=history, paper_path=paper
    )

    assert detail["status"] == "ok"
    assert detail["group_id"] == "exact"
