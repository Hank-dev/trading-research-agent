import json
from pathlib import Path

from trading_research_agent.schemas.portfolio import PortfolioFamily
from trading_research_agent.workflows import portfolio_batch as pb


def test_load_portfolio_batch_applies_defaults_and_aliases(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {
                    "family": "cross_sectional_momentum",
                    "lookback": 126,
                    "top_k": 2,
                    "rebalance": 21,
                    "start": "2010-01-01",
                    "end": "2025-01-01",
                },
                "portfolios": [
                    {"name": "Classic", "assets": ["SPY", "TLT", "DBC", "GLD"]},
                    {"assets": "SPY,QQQ,TLT,IEF,GLD", "top_k": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = pb.load_portfolio_batch(path)

    assert len(specs) == 2
    assert specs[0].name == "Classic"
    assert specs[0].assets == ["SPY", "TLT", "DBC", "GLD"]
    assert specs[0].portfolio_family == PortfolioFamily.CROSS_SECTIONAL_MOMENTUM
    assert specs[0].lookback_days == 126
    assert specs[0].top_k == 2
    assert specs[1].assets == ["SPY", "QQQ", "TLT", "IEF", "GLD"]
    assert specs[1].top_k == 1


def test_run_portfolio_batch_runs_each_spec(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(
        json.dumps(
            [
                {
                    "assets": ["SPY", "TLT"],
                    "family": "dual_momentum",
                    "start": "2015-01-01",
                    "end": "2025-01-01",
                },
                {
                    "assets": ["GLD", "DBC"],
                    "family": "time_series_momentum",
                    "start": "2015-01-01",
                    "end": "2025-01-01",
                },
            ]
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_portfolio_spec(spec, user_request, lockbox_pct=0.0):
        calls.append((spec.assets, user_request, lockbox_pct))
        return {"candidates": [], "winner_index": None, "winner_reason": "fake"}

    monkeypatch.setattr(pb, "run_portfolio_spec", fake_run_portfolio_spec)

    result = pb.run_portfolio_batch(path, lockbox_pct=0.2)

    assert result["count"] == 2
    assert len(result["results"]) == 2
    assert calls[0][0] == ["SPY", "TLT"]
    assert calls[0][2] == 0.2
    assert "Batch portfolio #1" in calls[0][1]
    assert calls[1][0] == ["GLD", "DBC"]
