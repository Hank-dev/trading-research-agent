"""Forward paper-trading: the only ungameable test.

A strategy can be tuned, re-cut, and stress-tested against historical data until
it looks good — but it cannot be fit to time that has not happened yet. This
module records a confirmed strategy as a dated paper position and, on demand,
replays it forward from its inception using whatever real data has accrued since,
reporting realized performance against the backtested expectation.

Inception defaults to the day the backtest ended — the exact boundary past which
every bar is genuinely out of sample.
"""

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any
import uuid

import pandas as pd

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    PortfolioVectorbtBackend,
    _import_vectorbt,
)
from trading_research_agent.config import get_output_path
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.data_loader import (
    load_fx_carry_rates,
    load_portfolio_panel,
)
from trading_research_agent.tools.portfolio_signals import compute_target_weights


def _aux_for(spec: PortfolioSpec, panel: pd.DataFrame) -> pd.DataFrame | None:
    """fx_carry needs a rate-differential panel aligned to the price index;
    other families are price-only."""
    if spec.portfolio_family == PortfolioFamily.FX_CARRY:
        return load_fx_carry_rates(
            spec.assets, spec.start_date, spec.end_date, panel.index
        )
    return None

PAPER_FILENAME = "paper_positions.jsonl"
PAPER_PATH = get_output_path(PAPER_FILENAME)
_MIN_FORWARD_DAYS = 63  # ~3 trading months before forward evidence means much


def default_paper_path() -> Path:
    return get_output_path(PAPER_FILENAME)


def append_paper_position(record: dict[str, Any], path: Path | None = None) -> None:
    path = path or default_paper_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_paper_positions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or default_paper_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def open_paper_position(
    winner: dict[str, Any], inception: str | None = None, path: Path | None = None
) -> dict[str, Any]:
    """Open a paper position from a reconstructed confirmed winner. Runs one
    full-period backtest to capture the expectation, then records the position."""
    from trading_research_agent.workflows.robustness_stress import spec_from_winner

    spec = spec_from_winner(winner)
    panel = load_portfolio_panel(spec.assets, spec.start_date, spec.end_date)
    result = PortfolioVectorbtBackend().run(spec, panel, _aux_for(spec, panel))
    m = result.metrics

    span_days = (
        date.fromisoformat(spec.end_date) - date.fromisoformat(spec.start_date)
    ).days
    inception_date = inception or spec.end_date  # default: where the backtest stopped

    record = {
        "id": uuid.uuid4().hex[:8],
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        "status": "open",
    }
    append_paper_position(record, path=path)
    return record


def _replay_forward(position: dict[str, Any], as_of: str) -> pd.Series:
    """Replay the strategy from a warmup start through `as_of`, returning the
    equity curve sliced to inception forward. Raises on bad data/spec."""
    params = position["params"]
    lookback = params["lookback_days"]
    inception = date.fromisoformat(position["inception_date"])
    warmup_start = (inception - timedelta(days=lookback * 2 + 30)).isoformat()

    spec = PortfolioSpec(
        name=f"paper-{position['id']}",
        assets=params["assets"],
        portfolio_family=PortfolioFamily(position["strategy_family"]),
        start_date=warmup_start,
        end_date=as_of,
        lookback_days=lookback,
        top_k=params.get("top_k", 1),
        rebalance_days=params["rebalance_days"],
        skip_recent_days=params.get("skip_recent_days", 252),
        hedge_weight=params.get("hedge_weight"),
        hypothesis="paper-trade forward replay",
    )
    panel = load_portfolio_panel(
        spec.assets, spec.start_date, spec.end_date, min_rows=lookback + 10
    )
    equity = _forward_equity(spec, panel)
    return equity[equity.index >= pd.Timestamp(inception)]


def forward_equity_series(position: dict[str, Any], as_of: str | None = None) -> pd.Series | None:
    """Best-effort forward equity curve for charting. Returns None on any failure."""
    as_of = as_of or date.today().isoformat()
    try:
        forward = _replay_forward(position, as_of)
    except Exception:
        return None
    return forward if len(forward) >= 2 else None


def evaluate_paper_position(position: dict[str, Any], as_of: str | None = None) -> dict[str, Any]:
    """Replay the position forward from inception to `as_of` (default today) using
    real accrued data, and compare realized performance to the expectation."""
    as_of = as_of or date.today().isoformat()

    try:
        forward = _replay_forward(position, as_of)
    except Exception as exc:
        return {"id": position["id"], "status": "error", "detail": str(exc)}

    if len(forward) < 2:
        return {
            "id": position["id"],
            "status": "no_data_yet",
            "inception_date": position["inception_date"],
            "as_of": as_of,
            "detail": "No bars have accrued since inception yet. Check back later.",
        }

    realized_return = float((forward.iloc[-1] / forward.iloc[0] - 1.0) * 100)
    running_max = forward.cummax()
    realized_max_dd = float(((forward / running_max - 1.0).min()) * 100)
    forward_days = (forward.index[-1] - forward.index[0]).days
    forward_trading_days = len(forward)
    realized_annualized = annualize(realized_return, forward_days)

    expectation = position["expectation"]
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
        "id": position["id"],
        "status": "evaluated",
        "inception_date": position["inception_date"],
        "as_of": str(forward.index[-1].date()),
        "forward_trading_days": forward_trading_days,
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
    if realized_max_dd < backtest_max_dd:
        return (
            "DRAWDOWN_BREACH",
            f"Live drawdown ({realized_max_dd:.1f}%) is already worse than the "
            f"backtest's worst ({backtest_max_dd:.1f}%). A serious red flag.",
        )
    if forward_trading_days < _MIN_FORWARD_DAYS:
        return (
            "TOO_EARLY",
            f"Only {forward_trading_days} forward trading days — too few to conclude "
            "anything. Forward evidence needs time to accrue.",
        )
    if realized_annualized < 0 < expected_annualized:
        return (
            "DIVERGING",
            f"Forward annualized return ({realized_annualized:.1f}%) is negative while "
            f"the backtest expected {expected_annualized:.1f}%. The edge is not showing up live.",
        )
    return (
        "TRACKING",
        f"Forward annualized {realized_annualized:.1f}% vs backtested {expected_annualized:.1f}%. "
        "Consistent so far, but keep accruing time before trusting it.",
    )


def _forward_equity(spec: PortfolioSpec, panel: pd.DataFrame) -> pd.Series:
    vbt = _import_vectorbt()
    weights = compute_target_weights(panel, spec, _aux_for(spec, panel))
    portfolio = vbt.Portfolio.from_orders(
        close=panel,
        size=weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=spec.initial_cash,
        fees=spec.commission_pct,
        slippage=spec.slippage_pct,
        freq="1D",
    )
    return portfolio.value()


def annualize(total_return_pct: float, span_days: int) -> float:
    if span_days <= 0:
        return 0.0
    base = max(1.0 + total_return_pct / 100.0, 1e-9)
    return float((base ** (365.0 / span_days) - 1.0) * 100.0)
