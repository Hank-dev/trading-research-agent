from trading_research_agent import app
from trading_research_agent.schemas.portfolio import PortfolioFamily
from trading_research_agent.schemas.report import ResearchReport


def test_portfolio_spec_cli_builds_exact_spec(monkeypatch) -> None:
    captured = {}

    def fake_run_portfolio_spec(spec, user_request, lockbox_pct=0.0):
        captured["spec"] = spec
        captured["user_request"] = user_request
        captured["lockbox_pct"] = lockbox_pct
        return {
            "candidates": [
                {
                    "strategy_spec": spec,
                    "report": ResearchReport(
                        markdown="# report",
                        verdict="needs_more_testing",
                        reasons=[],
                        next_tests=[],
                    ),
                }
            ],
            "winner_index": 0,
            "winner_reason": "hand_specified_strategy",
            "failure_summary": {
                "verdict_counts": {"needs_more_testing": 1},
                "failed_check_counts": {},
                "candidates_with_backtest": 0,
                "candidates_without_backtest": 1,
            },
        }

    monkeypatch.setattr(app, "run_portfolio_spec", fake_run_portfolio_spec)
    monkeypatch.setattr(app, "_save_exploration_reports", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_log_exploration_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app,
        "_refresh_dashboard_safely",
        lambda *_args, **_kwargs: captured.setdefault("dashboard_refreshed", True),
    )

    code = app.main(
        [
            "--portfolio-spec",
            "--assets",
            "GLD,SPY,USO,TLT,BTC-USD",
            "--family",
            "cross_sectional_momentum",
            "--lookback",
            "126",
            "--top-k",
            "2",
            "--rebalance",
            "21",
            "--start",
            "2015-01-01",
            "--end",
            "2026-05-29",
            "--lockbox-pct",
            "0.2",
        ]
    )

    assert code == 0
    spec = captured["spec"]
    assert spec.assets == ["GLD", "SPY", "USO", "TLT", "BTC-USD"]
    assert spec.portfolio_family == PortfolioFamily.CROSS_SECTIONAL_MOMENTUM
    assert spec.lookback_days == 126
    assert spec.top_k == 2
    assert spec.rebalance_days == 21
    assert spec.start_date == "2015-01-01"
    assert spec.end_date == "2026-05-29"
    assert captured["lockbox_pct"] == 0.2
    assert "Hand-specified portfolio" in captured["user_request"]
    assert captured["dashboard_refreshed"] is True


def test_data_health_cli_runs_preflight(monkeypatch) -> None:
    captured = {}

    def fake_check_data_health(assets, start, end):
        captured["assets"] = assets
        captured["start"] = start
        captured["end"] = end
        return {
            "assets": assets,
            "start": start,
            "end": end,
            "checks": [],
            "common_rows": 0,
            "common_start": None,
            "common_end": None,
            "min_rows": 300,
            "runnable": True,
            "reason": "OK",
        }

    monkeypatch.setattr(
        "trading_research_agent.workflows.data_health.check_data_health",
        fake_check_data_health,
    )

    code = app.main(
        [
            "--data-health",
            "--assets",
            "SPY,TLT,GLD",
            "--start",
            "2010-01-01",
            "--end",
            "2025-01-01",
        ]
    )

    assert code == 0
    assert captured == {
        "assets": ["SPY", "TLT", "GLD"],
        "start": "2010-01-01",
        "end": "2025-01-01",
    }


def test_portfolio_batch_cli_runs_file_and_refreshes_dashboard(monkeypatch) -> None:
    captured = {}

    def fake_run_portfolio_batch(path, lockbox_pct=0.0):
        captured["path"] = path
        captured["lockbox_pct"] = lockbox_pct
        return {
            "path": path,
            "lockbox_pct": lockbox_pct,
            "count": 1,
            "results": [
                {
                    "index": 1,
                    "spec": type("Spec", (), {"name": "Batch One"})(),
                    "user_request": "batch request",
                    "result": {
                        "candidates": [],
                        "winner_index": None,
                        "winner_reason": "fake",
                    },
                }
            ],
            "errors": [],
        }

    monkeypatch.setattr(app, "run_portfolio_batch", fake_run_portfolio_batch)
    monkeypatch.setattr(app, "_save_exploration_reports", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_log_exploration_history", lambda *_args, **_kwargs: "run123")
    monkeypatch.setattr(
        app,
        "_refresh_dashboard_safely",
        lambda *_args, **_kwargs: captured.setdefault("dashboard_refreshed", True),
    )

    code = app.main(["--portfolio-batch", "batch.json", "--lockbox-pct", "0.2"])

    assert code == 0
    assert captured["path"] == "batch.json"
    assert captured["lockbox_pct"] == 0.2
    assert captured["dashboard_refreshed"] is True


def test_mine_anomalies_cli_runs_miner(monkeypatch) -> None:
    captured = {}

    def fake_mine_anomalies(assets, start, end, top_n=12):
        captured["assets"] = assets
        captured["start"] = start
        captured["end"] = end
        captured["top_n"] = top_n
        return {"assets": assets, "start": start, "end": end, "facts": []}

    monkeypatch.setattr(
        "trading_research_agent.workflows.anomaly_miner.mine_anomalies",
        fake_mine_anomalies,
    )
    monkeypatch.setattr(
        "trading_research_agent.workflows.anomaly_miner.format_anomaly_report",
        lambda result: "anomaly report",
    )

    code = app.main(
        [
            "--mine-anomalies",
            "--assets",
            "SPY,TLT,GLD",
            "--start",
            "2010-01-01",
            "--end",
            "2024-12-31",
            "--top-anomalies",
            "5",
        ]
    )

    assert code == 0
    assert captured == {
        "assets": ["SPY", "TLT", "GLD"],
        "start": "2010-01-01",
        "end": "2024-12-31",
        "top_n": 5,
    }


def test_history_detail_cli_prints_without_rerunning(monkeypatch) -> None:
    captured = {}

    def fake_build_history_detail(identifier):
        captured["identifier"] = identifier
        return {"status": "ok"}

    monkeypatch.setattr(
        "trading_research_agent.workflows.history_detail.build_history_detail",
        fake_build_history_detail,
    )
    monkeypatch.setattr(
        "trading_research_agent.workflows.history_detail.render_history_detail_markdown",
        lambda detail: "# detail",
    )

    code = app.main(["--history-detail", "SPY,TLT,DBC,GLD"])

    assert code == 0
    assert captured["identifier"] == "SPY,TLT,DBC,GLD"
