"""Combined-book evaluation: does a hedge overlay improve the core, or just cost?

A standalone hedge backtest answers the wrong question — a hedge usually loses
money in isolation yet can still be worth holding if a small slice of it turns a
deep portfolio drawdown into a shallow one. This workflow answers the right
question: it runs the COMBINED book (core + overlay) and the CORE ALONE over the
same window and compares their drawdown, Sharpe, and return.

Verdict:
- IMPROVES_RISK_ADJUSTED: combined Sharpe is strictly higher than core-alone.
  The overlay is a genuine free improvement — rare and valuable.
- REDUCES_DRAWDOWN_AT_COST: combined Sharpe is not higher, but the drawdown is
  materially shallower. A judgment call: you pay return for tail protection.
- NOT_WORTH_IT: the overlay dragged returns without enough drawdown benefit.
"""

from typing import Any, TypedDict

import numpy as np
import pandas as pd

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    _import_vectorbt,
    _safe_float,
    _safe_optional_float,
)
from trading_research_agent.schemas.combined_book import CombinedBookSpec
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.tools.indicators import sma

_DRAWDOWN_IMPROVEMENT_RATIO = 0.80  # combined DD <= 80% of core DD depth = material


class BookMetrics(TypedDict):
    total_return_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    final_equity: float


class CombinedBookResult(TypedDict, total=False):
    spec_name: str
    core_assets: list[str]
    overlay_assets: list[str]
    overlay_weight: float
    overlay_rule: str
    core: BookMetrics
    combined: BookMetrics
    comparison: dict[str, Any]


def run_combined_book_eval(
    spec: CombinedBookSpec, panel: pd.DataFrame | None = None
) -> CombinedBookResult:
    if panel is None:
        from trading_research_agent.tools.data_loader import load_portfolio_panel

        panel = load_portfolio_panel(spec.all_assets(), spec.start_date, spec.end_date)

    panel = panel[[a for a in spec.all_assets() if a in panel.columns]]
    # Slice to the spec's own window so a pre-loaded master panel can be reused
    # across train and lockbox sub-periods.
    start = pd.Timestamp(spec.start_date)
    end = pd.Timestamp(spec.end_date)
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)].dropna(how="any")
    min_rows = spec.lookback_days + spec.rebalance_days + 5
    if len(panel) < min_rows:
        raise ValueError(
            f"Only {len(panel)} rows in {spec.start_date}..{spec.end_date}; "
            f"need at least {min_rows} for this lookback/rebalance."
        )

    combined_weights = compute_book_weights(panel, spec, include_overlay=True)
    core_weights = compute_book_weights(panel, spec, include_overlay=False)

    combined = _run_book(spec, panel, combined_weights)
    core = _run_book(spec, panel, core_weights)

    return {
        "spec_name": spec.name,
        "core_assets": list(spec.core_assets),
        "overlay_assets": list(spec.overlay_assets),
        "overlay_weight": spec.overlay_weight,
        "overlay_rule": spec.overlay_rule,
        "core": core,
        "combined": combined,
        "comparison": _compare(core, combined),
    }


def run_combined_book_with_lockbox(
    spec: CombinedBookSpec, lockbox_pct: float, panel: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Evaluate the overlay on a train segment, then re-check on a held-out tail
    the comparison never saw. `confirmed` is True only if the overlay still
    provides a benefit (verdict is not NOT_WORTH_IT) out of sample."""
    if lockbox_pct <= 0:
        return {"full": run_combined_book_eval(spec, panel)}

    if panel is None:
        from trading_research_agent.tools.data_loader import load_portfolio_panel

        panel = load_portfolio_panel(spec.all_assets(), spec.start_date, spec.end_date)

    train_end, lockbox_start = split_date_range(spec.start_date, spec.end_date, lockbox_pct)
    train_spec = spec.model_copy(update={"start_date": spec.start_date, "end_date": train_end})
    lockbox_spec = spec.model_copy(
        update={"start_date": lockbox_start, "end_date": spec.end_date}
    )

    out: dict[str, Any] = {
        "lockbox_split": {
            "original_start": spec.start_date,
            "original_end": spec.end_date,
            "train_end": train_end,
            "lockbox_start": lockbox_start,
        }
    }
    out["train"] = _safe_eval(train_spec, panel)
    out["lockbox"] = _safe_eval(lockbox_spec, panel)

    lockbox_result = out["lockbox"]
    out["confirmed"] = bool(
        lockbox_result is not None
        and lockbox_result["comparison"]["verdict"] != "NOT_WORTH_IT"
    )
    return out


def _safe_eval(spec: CombinedBookSpec, panel: pd.DataFrame) -> CombinedBookResult | None:
    try:
        return run_combined_book_eval(spec, panel)
    except Exception:
        return None


def compute_book_weights(
    panel: pd.DataFrame, spec: CombinedBookSpec, *, include_overlay: bool
) -> pd.DataFrame:
    """Target-weight matrix for either the combined book or the core alone. Both
    start at `lookback_days` and rebalance on the same cadence for a fair compare."""
    weights = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    n_core = len(spec.core_assets)
    n_overlay = len(spec.overlay_assets)

    for i in range(spec.lookback_days, len(panel), spec.rebalance_days):
        target = pd.Series(0.0, index=panel.columns)
        if not include_overlay:
            for asset in spec.core_assets:
                target[asset] = 1.0 / n_core
        elif spec.overlay_rule == "static":
            for asset in spec.core_assets:
                target[asset] = (1.0 - spec.overlay_weight) / n_core
            for asset in spec.overlay_assets:
                target[asset] = spec.overlay_weight / n_overlay
        else:  # regime: only carry the overlay when the core trend is down
            if _core_trend_up(panel, spec, i):
                for asset in spec.core_assets:
                    target[asset] = 1.0 / n_core
            else:
                for asset in spec.core_assets:
                    target[asset] = (1.0 - spec.overlay_weight) / n_core
                for asset in spec.overlay_assets:
                    target[asset] = spec.overlay_weight / n_overlay
        weights.iloc[i] = target.values

    return weights.shift(1)


def _core_trend_up(panel: pd.DataFrame, spec: CombinedBookSpec, i: int) -> bool:
    core = panel[spec.core_assets]
    composite = core.div(core.iloc[0]).mean(axis=1)
    sma_now = sma(composite.iloc[: i + 1], spec.lookback_days).iloc[-1]
    return bool(pd.notna(sma_now) and composite.iloc[i] > sma_now)


def _run_book(spec: CombinedBookSpec, panel: pd.DataFrame, weights: pd.DataFrame) -> BookMetrics:
    vbt = _import_vectorbt()
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
    return {
        "total_return_pct": _safe_float(portfolio.total_return()) * 100,
        "sharpe_ratio": _safe_optional_float(portfolio.sharpe_ratio()),
        "max_drawdown_pct": _safe_float(portfolio.max_drawdown()) * 100,
        "final_equity": _safe_float(portfolio.final_value()),
    }


def _compare(core: BookMetrics, combined: BookMetrics) -> dict[str, Any]:
    return_cost = core["total_return_pct"] - combined["total_return_pct"]
    drawdown_improvement = combined["max_drawdown_pct"] - core["max_drawdown_pct"]
    core_sharpe = core["sharpe_ratio"] or 0.0
    combined_sharpe = combined["sharpe_ratio"] or 0.0
    sharpe_delta = combined_sharpe - core_sharpe

    core_dd = core["max_drawdown_pct"]
    materially_shallower = (
        core_dd < 0 and (combined["max_drawdown_pct"] / core_dd) <= _DRAWDOWN_IMPROVEMENT_RATIO
    )

    if sharpe_delta > 0:
        verdict = "IMPROVES_RISK_ADJUSTED"
    elif materially_shallower:
        verdict = "REDUCES_DRAWDOWN_AT_COST"
    else:
        verdict = "NOT_WORTH_IT"

    return {
        "verdict": verdict,
        "return_cost_pct": return_cost,
        "drawdown_improvement_pct": drawdown_improvement,
        "sharpe_delta": sharpe_delta,
    }
