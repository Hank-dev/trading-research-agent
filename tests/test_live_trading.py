from pathlib import Path

import pandas as pd
import pytest

from trading_research_agent.workflows import live_trading as lt


def make_book(**overrides) -> dict:
    data = {
        "book_id": "book1",
        "created_at": "2026-01-01T00:00:00Z",
        "origin_timestamp": "2026-01-01T00:00:00Z",
        "inception_date": "2026-01-03",
        "strategy_family": "cross_sectional_momentum",
        "params": {
            "assets": ["AAA", "BBB"],
            "lookback_days": 20,
            "top_k": 1,
            "rebalance_days": 5,
            "skip_recent_days": 252,
            "hedge_weight": None,
        },
        "initial_cash": 10_000.0,
        "commission_pct": 0.0,
        "slippage_pct": 0.0,
        "expectation": {
            "annualized_return_pct": 12.0,
            "backtest_total_return_pct": 80.0,
            "backtest_span_days": 3650,
            "backtest_sharpe": 1.0,
            "backtest_max_drawdown_pct": -20.0,
        },
        "cash": 10_000.0,
        "positions": {"AAA": 0.0, "BBB": 0.0},
        "last_bar_date": None,
        "status": "open",
        "ledger": [],
    }
    data.update(overrides)
    return data


def test_save_load_and_list_books_roundtrip(tmp_path: Path) -> None:
    first = make_book(book_id="first", inception_date="2026-01-03")
    second = make_book(book_id="second", inception_date="2026-01-02")
    closed = make_book(book_id="closed", status="closed", inception_date="2026-01-01")

    lt.save_book(first, root=tmp_path)
    lt.save_book(second, root=tmp_path)
    lt.save_book(closed, root=tmp_path)

    assert lt.load_book("first", root=tmp_path)["book_id"] == "first"
    assert [b["book_id"] for b in lt.list_books(root=tmp_path)] == [
        "closed",
        "second",
        "first",
    ]
    assert [b["book_id"] for b in lt.list_open_books(root=tmp_path)] == [
        "second",
        "first",
    ]


def test_execute_bar_sizes_targets_after_costs_without_negative_cash() -> None:
    book = make_book(commission_pct=0.001, slippage_pct=0.0005)
    close = pd.Series({"AAA": 100.0, "BBB": 100.0})

    entry = lt._execute_bar(
        book=book,
        bar_date=pd.Timestamp("2026-01-03"),
        assets=["AAA", "BBB"],
        close=close,
        target_weights={"AAA": 1.0, "BBB": 0.0},
        cost_rate=0.0015,
    )

    expected_nav = 10_000.0 / 1.0015
    assert entry["post_nav"] == pytest.approx(expected_nav, abs=1e-3)
    assert entry["costs"] == pytest.approx(expected_nav * 0.0015, abs=1e-3)
    assert book["cash"] == pytest.approx(0.0, abs=1e-8)
    assert book["positions"]["AAA"] == pytest.approx(expected_nav / 100.0)
    assert book["positions"]["BBB"] == 0.0


def test_tick_writes_daily_mark_to_market_rows_and_is_idempotent(
    monkeypatch, tmp_path: Path
) -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    panel = pd.DataFrame(
        {
            "AAA": [100.0, 100.0, 100.0, 110.0, 120.0],
            "BBB": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    weights = pd.DataFrame(float("nan"), index=dates, columns=["AAA", "BBB"])
    weights.loc[pd.Timestamp("2026-01-03")] = [1.0, 0.0]

    monkeypatch.setattr(
        lt,
        "load_portfolio_panel",
        lambda assets, start, end, min_rows=30: panel,
    )
    monkeypatch.setattr(lt, "compute_target_weights", lambda panel_arg, spec, aux: weights)

    book = make_book()
    updated = lt.tick_live_book(book, as_of="2026-01-05", root=tmp_path)

    assert [row["date"] for row in updated["ledger"]] == [
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
    ]
    assert [row["traded_value"] for row in updated["ledger"]] == [
        10_000.0,
        0.0,
        0.0,
    ]
    assert [row["post_nav"] for row in updated["ledger"]] == [
        10_000.0,
        11_000.0,
        12_000.0,
    ]
    assert updated["last_bar_date"] == "2026-01-05"

    second = lt.tick_live_book(updated, as_of="2026-01-05", root=tmp_path)
    assert len(second["ledger"]) == 3
    assert lt.load_book("book1", root=tmp_path)["last_bar_date"] == "2026-01-05"


def test_evaluate_live_book_reports_current_nav() -> None:
    book = make_book(
        cash=0.0,
        positions={"AAA": 100.0, "BBB": 0.0},
        last_bar_date="2026-01-05",
        ledger=[
            {"date": "2026-01-03", "post_nav": 10_000.0},
            {"date": "2026-01-04", "post_nav": 11_000.0},
            {"date": "2026-01-05", "post_nav": 12_000.0},
        ],
    )

    ev = lt.evaluate_live_book(book)

    assert ev["status"] == "evaluated"
    assert ev["nav"] == 12_000.0
    assert ev["realized_return_pct"] == pytest.approx(20.0)
    assert ev["forward_trading_days"] == 3
    assert ev["read"] == "TOO_EARLY"


def test_auto_promote_opens_book_for_newer_confirmed_winner(monkeypatch, tmp_path: Path) -> None:
    from trading_research_agent.workflows import robustness_stress as rs

    current = make_book(book_id="current", origin_timestamp="2026-01-01T00:00:00Z")
    lt.save_book(current, root=tmp_path)
    winner = {
        "timestamp": "2026-02-01T00:00:00Z",
        "strategy_family": "cross_sectional_momentum",
        "params": {"assets": ["AAA", "BBB"]},
        "full_start": "2020-01-01",
        "full_end": "2026-01-01",
    }
    monkeypatch.setattr(rs, "latest_confirmed_portfolio_winner", lambda history: winner)

    def fake_open_live_book(winner_arg, *, inception=None, root=None):
        book = make_book(
            book_id="newbook",
            origin_timestamp=winner_arg["timestamp"],
            inception_date=inception or winner_arg["full_end"],
        )
        lt.save_book(book, root=root)
        return book

    monkeypatch.setattr(lt, "open_live_book", fake_open_live_book)

    result = lt.auto_promote([{"dummy": True}], root=tmp_path)

    assert result["action"] == "promoted"
    assert result["opened"] == ["newbook"]
    assert lt.load_book("newbook", root=tmp_path)["origin_timestamp"] == winner["timestamp"]
