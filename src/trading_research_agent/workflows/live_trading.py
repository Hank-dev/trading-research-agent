"""Live paper-trading engine: bar-by-bar, stateful position tracking.

The existing ``paper_trading`` module *re-replays* the entire forward window
through vectorbt every time you call ``--paper-status``. That is great for a
"did reality match the backtest?" equity curve, but it keeps no real state: it
has no per-trade ledger, no "today's order", and no cash/positions that persist
between calls.

This module is the bridge from forward replay to stateful paper trading. It
maintains a **persistent book** (cash + shares + a per-bar ledger) and advances
one trading day at a time via :func:`tick_live_book`. Each tick:

1. Loads the warmup price panel (+ fx-carry aux if needed) ending at the latest
   available bar <= ``as_of``.
2. Computes target weights for every new bar since ``last_bar_date`` using the
   *same* :func:`compute_target_weights` the backtest uses — so the live signal
   is definitionally identical to the backtest signal, never a re-implementation.
3. Executes each new bar at its close: rebalances to the target weight, books
   commission + slippage, updates cash and positions, and appends a ledger row.
4. Persists the updated book.

Idempotent by design: calling tick twice in one day is a no-op because
``last_bar_date`` advances past the processed bar. Safe to drive from a daily
cron/systemd timer.

Auto-promotion: :func:`find_newer_confirmed_winner` checks research history for
a lockbox-confirmed winner fresher than the book's origin and, if found, opens
a new parallel book — so the live book tracks your best current understanding
without discarding the running experiment.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    PortfolioVectorbtBackend,
)
from trading_research_agent.config import get_output_path
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.data_loader import (
    load_fx_carry_rates,
    load_portfolio_panel,
)
from trading_research_agent.tools.portfolio_signals import compute_target_weights
from trading_research_agent.workflows.paper_trading import annualize

LIVE_BOOKS_DIR = get_output_path("live_books")


# ---------------------------------------------------------------------------
# Paths / I/O
# ---------------------------------------------------------------------------


def default_live_books_dir() -> Path:
    return LIVE_BOOKS_DIR


def _book_path(book_id: str, root: Path | None = None) -> Path:
    return (root or LIVE_BOOKS_DIR) / f"{book_id}.json"


def save_book(book: dict[str, Any], root: Path | None = None) -> Path:
    """Atomically persist a live book to disk. Returns the path written."""
    path = _book_path(book["book_id"], root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(book, sort_keys=True, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_book(book_id: str, root: Path | None = None) -> dict[str, Any] | None:
    path = _book_path(book_id, root=root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_books(root: Path | None = None) -> list[dict[str, Any]]:
    """All live books, ordered by inception date ascending."""
    base = root or LIVE_BOOKS_DIR
    if not base.exists():
        return []
    books: list[dict[str, Any]] = []
    for p in sorted(base.glob("*.json")):
        try:
            books.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    books.sort(key=lambda b: b.get("inception_date", ""))
    return books


def list_open_books(root: Path | None = None) -> list[dict[str, Any]]:
    return [b for b in list_books(root=root) if b.get("status") == "open"]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Opening a book
# ---------------------------------------------------------------------------


def open_live_book(
    winner: dict[str, Any],
    *,
    inception: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Open a persistent live book from a reconstructed confirmed winner.

    Runs one full-period backtest to capture the expectation (identical to
    :func:`paper_trading.open_paper_position`), then initialises the state at
    all-cash with an empty ledger. The first tick fills in the first bar.
    """
    from trading_research_agent.workflows.robustness_stress import spec_from_winner

    spec = spec_from_winner(winner)
    panel = load_portfolio_panel(spec.assets, spec.start_date, spec.end_date)
    aux = _aux_for(spec, panel)
    result = PortfolioVectorbtBackend().run(spec, panel, aux)
    m = result.metrics

    span_days = (
        date.fromisoformat(spec.end_date) - date.fromisoformat(spec.start_date)
    ).days
    inception_date = inception or spec.end_date

    book = {
        "book_id": uuid.uuid4().hex[:8],
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "origin_timestamp": winner.get("timestamp", ""),
        "inception_date": inception_date,
        "strategy_family": spec.portfolio_family.value,
        "params": {
            "assets": list(spec.assets),
            "lookback_days": spec.lookback_days,
            "top_k": spec.top_k,
            "rebalance_days": spec.rebalance_days,
            "skip_recent_days": spec.skip_recent_days,
            "hedge_weight": spec.hedge_weight,
        },
        "initial_cash": spec.initial_cash,
        "commission_pct": spec.commission_pct,
        "slippage_pct": spec.slippage_pct,
        "expectation": {
            "annualized_return_pct": annualize(m.total_return_pct, span_days),
            "backtest_total_return_pct": m.total_return_pct,
            "backtest_span_days": span_days,
            "backtest_sharpe": m.sharpe_ratio,
            "backtest_max_drawdown_pct": m.max_drawdown_pct,
        },
        # Live state — mutated by each tick.
        "cash": float(spec.initial_cash),
        "positions": {asset: 0.0 for asset in spec.assets},
        "last_bar_date": None,
        "status": "open",
        "ledger": [],
    }
    save_book(book, root=root)
    return book


# ---------------------------------------------------------------------------
# Ticking — the engine core
# ---------------------------------------------------------------------------


def tick_live_book(
    book: dict[str, Any],
    *,
    as_of: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Advance the book by every trading day between ``last_bar_date`` and ``as_of``.

    ``as_of`` defaults to today. For each new bar the book rebalances to the
    strategy's target weight at that bar's close, booking commission + slippage,
    and appends a ledger row. Idempotent: a second call with no new bars is a
    no-op. Returns the book (mutated in place and persisted).
    """
    as_of = as_of or date.today().isoformat()
    params = book["params"]
    assets: list[str] = params["assets"]
    lookback = params["lookback_days"]

    warmup_start = _warmup_start(book["inception_date"], lookback)
    spec = _build_live_spec(book, warmup_start, as_of)

    panel = load_portfolio_panel(assets, spec.start_date, spec.end_date, min_rows=lookback + 10)
    aux = _aux_for(spec, panel)
    weights = compute_target_weights(panel, spec, aux)

    # Only bars at/after inception and strictly after the last processed bar.
    as_of_ts = pd.Timestamp(as_of)
    inception_ts = pd.Timestamp(book["inception_date"])
    last_ts = pd.Timestamp(book["last_bar_date"]) if book["last_bar_date"] else None

    candidate_dates = panel.index[panel.index <= as_of_ts]
    candidate_dates = candidate_dates[candidate_dates >= inception_ts]
    if last_ts is not None:
        candidate_dates = candidate_dates[candidate_dates > last_ts]

    cost_rate = float(book["commission_pct"]) + float(book["slippage_pct"])

    for bar_date in candidate_dates:
        weight_row = weights.loc[bar_date]
        close_row = panel.loc[bar_date]
        if weight_row.isna().all():
            entry = _mark_to_market_bar(
                book=book,
                bar_date=bar_date,
                assets=assets,
                close=close_row,
            )
        else:
            target_weights = _validated_target_weights(weight_row, assets)
            entry = _execute_bar(
                book=book,
                bar_date=bar_date,
                assets=assets,
                close=close_row,
                target_weights=target_weights,
                cost_rate=cost_rate,
            )
        book["ledger"].append(entry)
        book["last_bar_date"] = bar_date.strftime("%Y-%m-%d")
        # cash and positions are updated inside the bar handler.

    save_book(book, root=root)
    return book


def _execute_bar(
    *,
    book: dict[str, Any],
    bar_date: pd.Timestamp,
    assets: list[str],
    close: pd.Series,
    target_weights: dict[str, float],
    cost_rate: float,
) -> dict[str, Any]:
    """Rebalance the book to ``target_weights`` at this bar's close.

    Execution model:
      - Mark the book to market at ``close`` to get pre-trade NAV.
      - Solve target_value = weight * post-cost NAV for all assets.
      - Move positions to target; cash absorbs the net trade value + costs.
      - post_nav = pre_nav - costs (rebalancing is zero-sum net of costs).
    """
    prices = _validated_prices(close, assets)
    current_values = {
        asset: float(book["positions"].get(asset, 0.0)) * prices[asset]
        for asset in assets
    }
    pre_nav = float(book["cash"]) + sum(current_values.values())
    if not np.isfinite(pre_nav) or pre_nav <= 0.0:
        raise ValueError(f"Book NAV must be positive before trading; got {pre_nav!r}")

    target_values, total_traded, total_cost, post_nav = _cost_adjusted_targets(
        pre_nav=pre_nav,
        current_values=current_values,
        target_weights=target_weights,
        cost_rate=cost_rate,
    )
    new_positions = {
        asset: target_values[asset] / prices[asset]
        for asset in assets
    }
    invested = sum(target_values.values())
    cash = post_nav - invested
    if abs(cash) < 1e-8:
        cash = 0.0
    if cash < -1e-6:
        raise ValueError(f"Rebalance would leave negative cash: {cash:.6f}")

    book["cash"] = float(cash)
    book["positions"] = {a: float(new_positions[a]) for a in assets}

    return {
        "date": bar_date.strftime("%Y-%m-%d"),
        "close": prices,
        "target_weights": {a: float(target_weights.get(a, 0.0)) for a in assets},
        "pre_nav": round(pre_nav, 6),
        "post_nav": round(post_nav, 6),
        "cash": round(book["cash"], 6),
        "positions": {a: round(book["positions"][a], 8) for a in assets},
        "traded_value": round(total_traded, 6),
        "costs": round(total_cost, 6),
    }


def _mark_to_market_bar(
    *,
    book: dict[str, Any],
    bar_date: pd.Timestamp,
    assets: list[str],
    close: pd.Series,
) -> dict[str, Any]:
    prices = _validated_prices(close, assets)
    post_nav = float(book["cash"]) + sum(
        float(book["positions"].get(asset, 0.0)) * prices[asset] for asset in assets
    )
    if not np.isfinite(post_nav) or post_nav < 0.0:
        raise ValueError(f"Book NAV is invalid while marking to market: {post_nav!r}")
    return {
        "date": bar_date.strftime("%Y-%m-%d"),
        "close": prices,
        "target_weights": _current_weights(book, assets, prices, post_nav),
        "pre_nav": round(post_nav, 6),
        "post_nav": round(post_nav, 6),
        "cash": round(float(book["cash"]), 6),
        "positions": {
            a: round(float(book["positions"].get(a, 0.0)), 8) for a in assets
        },
        "traded_value": 0.0,
        "costs": 0.0,
    }


def _cost_adjusted_targets(
    *,
    pre_nav: float,
    current_values: dict[str, float],
    target_weights: dict[str, float],
    cost_rate: float,
) -> tuple[dict[str, float], float, float, float]:
    """Solve target values after transaction costs.

    Target-percent orders are specified as fractions of post-cost NAV. Because
    costs depend on trade size and trade size depends on post-cost NAV, solve the
    small fixed-point problem directly. This keeps cash non-negative for fully
    invested books instead of buying the pre-cost notional and borrowing fees.
    """
    post_nav = pre_nav
    total_traded = 0.0
    total_cost = 0.0
    target_values = {asset: 0.0 for asset in current_values}

    for _ in range(50):
        target_values = {
            asset: float(target_weights.get(asset, 0.0)) * post_nav
            for asset in current_values
        }
        total_traded = sum(
            abs(target_values[asset] - current_values[asset])
            for asset in current_values
        )
        total_cost = total_traded * cost_rate
        next_post_nav = pre_nav - total_cost
        if next_post_nav < 0.0:
            raise ValueError("Transaction costs exceed book NAV")
        if abs(next_post_nav - post_nav) <= 1e-9:
            post_nav = next_post_nav
            break
        post_nav = next_post_nav

    target_values = {
        asset: float(target_weights.get(asset, 0.0)) * post_nav
        for asset in current_values
    }
    total_traded = sum(
        abs(target_values[asset] - current_values[asset])
        for asset in current_values
    )
    total_cost = total_traded * cost_rate
    post_nav = pre_nav - total_cost
    return target_values, total_traded, total_cost, post_nav


def _validated_target_weights(row: pd.Series, assets: list[str]) -> dict[str, float]:
    asset_row = row.reindex(assets)
    if asset_row.isna().any():
        missing = [asset for asset in assets if pd.isna(asset_row.get(asset))]
        raise ValueError(f"Partial target weights for {missing}; expected all assets")

    weights: dict[str, float] = {}
    for asset in assets:
        value = float(asset_row[asset])
        if not np.isfinite(value):
            raise ValueError(f"Invalid target weight for {asset}: {value!r}")
        if value < -1e-12:
            raise ValueError(f"Negative target weight for {asset}: {value:.6f}")
        weights[asset] = 0.0 if abs(value) < 1e-12 else value
    total = sum(weights.values())
    if total > 1.000001:
        raise ValueError(f"Target weights sum to {total:.6f}; expected <= 1.0")
    return weights


def _validated_prices(close: pd.Series, assets: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for asset in assets:
        price = float(close[asset])
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError(f"Invalid close price for {asset}: {price!r}")
        prices[asset] = price
    return prices


def _current_weights(
    book: dict[str, Any],
    assets: list[str],
    prices: dict[str, float],
    nav: float,
) -> dict[str, float]:
    if nav <= 0.0:
        return {asset: 0.0 for asset in assets}
    return {
        asset: float(book["positions"].get(asset, 0.0)) * prices[asset] / nav
        for asset in assets
    }


# ---------------------------------------------------------------------------
# Evaluation / status
# ---------------------------------------------------------------------------


def evaluate_live_book(book: dict[str, Any]) -> dict[str, Any]:
    """Summarise a live book's realised performance vs the backtest expectation."""
    ledger = book.get("ledger", [])
    if not ledger:
        return {
            "book_id": book["book_id"],
            "status": "no_bars_yet",
            "inception_date": book["inception_date"],
            "detail": "No bars have been ticked yet. Run --live-tick after market close.",
        }

    equity = pd.Series(
        [row["post_nav"] for row in ledger],
        index=pd.to_datetime([row["date"] for row in ledger]),
    )
    realized_return = float((equity.iloc[-1] / book["initial_cash"] - 1.0) * 100)
    running_max = equity.cummax()
    realized_max_dd = float(((equity / running_max - 1.0).min()) * 100)
    forward_days = (equity.index[-1] - equity.index[0]).days
    forward_trading_days = len(equity)
    realized_annualized = annualize(realized_return, forward_days)

    expectation = book["expectation"]
    expected_annualized = expectation["annualized_return_pct"]
    backtest_max_dd = expectation["backtest_max_drawdown_pct"]

    read, detail = _read(
        forward_trading_days=forward_trading_days,
        realized_annualized=realized_annualized,
        expected_annualized=expected_annualized,
        realized_max_dd=realized_max_dd,
        backtest_max_dd=backtest_max_dd,
    )

    return {
        "book_id": book["book_id"],
        "status": "evaluated",
        "strategy_family": book["strategy_family"],
        "assets": book["params"]["assets"],
        "inception_date": book["inception_date"],
        "as_of": str(equity.index[-1].date()),
        "forward_trading_days": forward_trading_days,
        "nav": float(equity.iloc[-1]),
        "cash": book["cash"],
        "positions": book["positions"],
        "realized_return_pct": realized_return,
        "realized_annualized_pct": realized_annualized,
        "realized_max_drawdown_pct": realized_max_dd,
        "expected_annualized_pct": expected_annualized,
        "backtest_max_drawdown_pct": backtest_max_dd,
        "read": read,
        "detail": detail,
    }


def _read(
    *,
    forward_trading_days: int,
    realized_annualized: float,
    expected_annualized: float,
    realized_max_dd: float,
    backtest_max_dd: float,
) -> tuple[str, str]:
    """Same rubric as paper_trading._read — kept here so live status is
    self-contained and the two modules can drift independently if needed."""
    if realized_max_dd < backtest_max_dd:
        return (
            "DRAWDOWN_BREACH",
            f"Live drawdown ({realized_max_dd:.1f}%) is already worse than the "
            f"backtest's worst ({backtest_max_dd:.1f}%).",
        )
    if forward_trading_days < 63:
        return (
            "TOO_EARLY",
            f"Only {forward_trading_days} live bars — too few to conclude anything. "
            "Keep ticking.",
        )
    if realized_annualized < 0 < expected_annualized:
        return (
            "DIVERGING",
            f"Live annualized ({realized_annualized:.1f}%) is negative while the "
            f"backtest expected {expected_annualized:.1f}%.",
        )
    return (
        "TRACKING",
        f"Live annualized {realized_annualized:.1f}% vs backtested "
        f"{expected_annualized:.1f}%. Consistent so far.",
    )


# ---------------------------------------------------------------------------
# Auto-promotion
# ---------------------------------------------------------------------------


def find_newer_confirmed_winner(
    history: list[dict[str, Any]],
    current_origin_timestamp: str,
) -> dict[str, Any] | None:
    """Return a confirmed portfolio winner newer than the book's origin, else None.

    Delegates to :func:`robustness_stress.latest_confirmed_portfolio_winner` so
    the "what counts as confirmed" definition stays in one place.
    """
    from trading_research_agent.workflows.robustness_stress import (
        latest_confirmed_portfolio_winner,
    )

    winner = latest_confirmed_portfolio_winner(history)
    if winner is None:
        return None
    if (winner.get("timestamp") or "") > (current_origin_timestamp or ""):
        return winner
    return None


def auto_promote(
    history: list[dict[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Open new live books for any confirmed winner fresher than every open book.

    Returns a summary dict. Each open book's origin is checked; if a fresher
    winner exists, a new parallel book is opened (the old book keeps running).
    """
    opened: list[str] = []
    open_books = list_open_books(root=root)
    if not open_books:
        return {"action": "no_open_books", "opened": opened}

    # The newest origin across all open books is the bar to beat.
    newest_origin = max(b.get("origin_timestamp", "") for b in open_books)
    winner = find_newer_confirmed_winner(history, newest_origin)
    if winner is None:
        return {"action": "no_new_winner", "opened": opened}

    book = open_live_book(winner, root=root)
    opened.append(book["book_id"])
    return {"action": "promoted", "opened": opened, "winner_ts": winner.get("timestamp")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aux_for(spec: PortfolioSpec, panel: pd.DataFrame) -> pd.DataFrame | None:
    """fx_carry needs a rate-differential panel; other families are price-only."""
    if spec.portfolio_family == PortfolioFamily.FX_CARRY:
        return load_fx_carry_rates(
            spec.assets, spec.start_date, spec.end_date, panel.index
        )
    return None


def _warmup_start(inception_date: str, lookback: int) -> str:
    inception = date.fromisoformat(inception_date)
    # ~2x lookback in calendar days + a 30-day buffer covers weekends/holidays.
    return (inception - timedelta(days=lookback * 2 + 30)).isoformat()


def _build_live_spec(
    book: dict[str, Any], warmup_start: str, as_of: str
) -> PortfolioSpec:
    """Construct a PortfolioSpec spanning warmup -> as_of for signal computation."""
    params = book["params"]
    return PortfolioSpec(
        name=f"live-{book['book_id']}",
        assets=params["assets"],
        portfolio_family=PortfolioFamily(book["strategy_family"]),
        start_date=warmup_start,
        end_date=as_of,
        lookback_days=params["lookback_days"],
        top_k=params.get("top_k", 1),
        rebalance_days=params["rebalance_days"],
        skip_recent_days=params.get("skip_recent_days", 252),
        hedge_weight=params.get("hedge_weight"),
        hypothesis="live paper-trade tick",
    )
