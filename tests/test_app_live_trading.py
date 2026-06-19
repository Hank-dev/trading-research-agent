from trading_research_agent import app


def make_book() -> dict:
    return {
        "book_id": "book1",
        "status": "open",
        "strategy_family": "cross_sectional_momentum",
        "inception_date": "2026-01-03",
        "last_bar_date": "2026-01-05",
        "params": {"assets": ["AAA", "BBB"]},
        "cash": 0.0,
        "positions": {"AAA": 100.0, "BBB": 0.0},
        "ledger": [{"date": "2026-01-05", "post_nav": 12_000.0}],
    }


def make_eval() -> dict:
    return {
        "book_id": "book1",
        "status": "evaluated",
        "strategy_family": "cross_sectional_momentum",
        "assets": ["AAA", "BBB"],
        "inception_date": "2026-01-03",
        "as_of": "2026-01-05",
        "forward_trading_days": 3,
        "nav": 12_000.0,
        "cash": 0.0,
        "positions": {"AAA": 100.0, "BBB": 0.0},
        "realized_return_pct": 20.0,
        "realized_annualized_pct": 100.0,
        "realized_max_drawdown_pct": 0.0,
        "expected_annualized_pct": 12.0,
        "backtest_max_drawdown_pct": -20.0,
        "read": "TOO_EARLY",
        "detail": "Too early.",
    }


def test_live_status_cli_evaluates_requested_book(monkeypatch) -> None:
    captured = {}
    book = make_book()

    def fake_load_book(book_id):
        captured["book_id"] = book_id
        return book

    monkeypatch.setattr(
        "trading_research_agent.workflows.live_trading.load_book",
        fake_load_book,
    )
    monkeypatch.setattr(
        "trading_research_agent.workflows.live_trading.evaluate_live_book",
        lambda book_arg: make_eval(),
    )

    code = app.main(["--live-status", "--book-id", "book1"])

    assert code == 0
    assert captured["book_id"] == "book1"


def test_live_tick_cli_passes_as_of_and_persists_updated_book(monkeypatch) -> None:
    captured = {}
    book = make_book()

    def fake_tick_live_book(book_arg, *, as_of=None):
        captured["as_of"] = as_of
        book_arg["ledger"].append({"date": as_of, "post_nav": 12_500.0})
        book_arg["last_bar_date"] = as_of
        return book_arg

    monkeypatch.setattr(
        "trading_research_agent.workflows.live_trading.load_book",
        lambda book_id: book,
    )
    monkeypatch.setattr(
        "trading_research_agent.workflows.live_trading.tick_live_book",
        fake_tick_live_book,
    )
    monkeypatch.setattr(
        "trading_research_agent.workflows.live_trading.evaluate_live_book",
        lambda book_arg: make_eval() | {"nav": 12_500.0, "as_of": "2026-01-06"},
    )

    code = app.main(["--live-tick", "--book-id", "book1", "--as-of", "2026-01-06"])

    assert code == 0
    assert captured["as_of"] == "2026-01-06"
    assert book["last_bar_date"] == "2026-01-06"


def test_live_flags_are_mutually_exclusive() -> None:
    code = None
    try:
        app.main(["--live-status", "--live-tick"])
    except SystemExit as exc:
        code = exc.code

    assert code == 2
