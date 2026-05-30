"""Macro-regime asset rotation — the disciplined, look-ahead-safe version.

A frozen, pre-registered strategy: a single monetary-liquidity signal (default
WALCL, the Fed balance sheet) defines a binary regime, and a mapping committed
BEFORE any backtest decides the allocation. There are no tunable mapping knobs on
the CLI — that is deliberate, because the freedom to re-map assets to regimes is
exactly the overfitting surface this guards against.

Honest expectations: there are only a handful of distinct liquidity regimes in
the data, so significance is hard to establish and the lockbox is likely to be
unkind. Finding that out cleanly is the point.

Pre-committed mapping (frozen):
- EXPANSION  (balance sheet higher than a quarter ago): equal-weight SPY, QQQ, GLD, USO
- CONTRACTION(balance sheet lower than a quarter ago):  100% TLT

Look-ahead guards: the macro series is lagged by its publication delay, the regime
compares today vs a quarter ago (both backward-looking), and target weights execute
on the next bar.
"""

from typing import Any, TypedDict

import numpy as np
import pandas as pd

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    _import_vectorbt,
    _safe_float,
    _safe_optional_float,
)
from trading_research_agent.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE_PCT,
)
from trading_research_agent.reports.markdown_report import build_research_report
from trading_research_agent.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    RobustnessResult,
)
from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.tools.portfolio_signals import equal_weight_benchmark_return_pct
from trading_research_agent.tools.stats import estimate_trading_days, probabilistic_sharpe_ratio

MACRO_SIGNALS: dict[str, dict[str, Any]] = {
    "walcl": {
        "series_id": "WALCL",
        "label": "Fed balance sheet (WALCL)",
        "publication_lag_days": 7,
        "change_window_days": 63,  # quarter-over-quarter change
    },
    "fedfunds": {
        "series_id": "FEDFUNDS",
        "label": "Fed funds rate (FEDFUNDS)",
        "publication_lag_days": 14,
        "change_window_days": 63,
    },
    "m2": {
        "series_id": "M2SL",
        "label": "M2 money stock (M2SL)",
        "publication_lag_days": 35,
        "change_window_days": 63,
    },
}

# Frozen, pre-registered mapping. Do not parameterize on the CLI.
ASSETS = ["SPY", "QQQ", "GLD", "USO", "TLT"]
EXPANSION_ASSETS = ["SPY", "QQQ", "GLD", "USO"]
CONTRACTION_ASSETS = ["TLT"]

_REBALANCE_DAYS = 21
_PSR_PASS = 0.95


class MacroRegimeResult(TypedDict, total=False):
    signal: str
    assets: list[str]
    full: dict[str, Any]
    train: dict[str, Any]
    lockbox: dict[str, Any]
    lockbox_split: dict[str, str]
    confirmed: bool


def run_macro_regime(signal_key: str, start: str, end: str, lockbox_pct: float) -> MacroRegimeResult:
    if signal_key not in MACRO_SIGNALS:
        raise ValueError(f"unknown macro signal {signal_key}; choose from {sorted(MACRO_SIGNALS)}")
    cfg = MACRO_SIGNALS[signal_key]

    from trading_research_agent.tools.data_loader import load_fred_series, load_portfolio_panel

    panel = load_portfolio_panel(ASSETS, start, end)
    # Load the macro series with a buffer before `start` so it is warm at the open.
    buffer_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).date().isoformat()
    signal = load_fred_series(cfg["series_id"], buffer_start, end)

    expansion = _regime_for_panel(signal, panel.index, cfg)
    weights = _build_weights(panel, expansion, cfg["change_window_days"])

    result: MacroRegimeResult = {
        "signal": cfg["label"],
        "assets": list(ASSETS),
        "full": _segment(panel, weights, start, end),
    }

    if lockbox_pct > 0:
        train_end, lockbox_start = split_date_range(start, end, lockbox_pct)
        result["lockbox_split"] = {
            "original_start": start,
            "original_end": end,
            "train_end": train_end,
            "lockbox_start": lockbox_start,
        }
        result["train"] = _segment(panel, weights, start, train_end)
        result["lockbox"] = _segment(panel, weights, lockbox_start, end)
        result["confirmed"] = result["lockbox"]["verdict"] == "worth_paper_trading"

    return result


def _regime_for_panel(signal: pd.Series, panel_index: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """Boolean expansion series aligned to panel dates, with no look-ahead.

    The value used on panel date D is the most recent macro observation on or
    before (D - publication_lag). Expansion = that value is higher than it was
    `change_window_days` trading days earlier.
    """
    lag = pd.Timedelta(days=cfg["publication_lag_days"])
    lagged_dates = panel_index - lag
    union = signal.index.union(pd.DatetimeIndex(lagged_dates))
    ffilled = signal.reindex(union).sort_index().ffill()
    known = pd.Series(ffilled.reindex(pd.DatetimeIndex(lagged_dates)).to_numpy(), index=panel_index)
    expansion = known > known.shift(cfg["change_window_days"])
    return expansion.fillna(False)


def _build_weights(panel: pd.DataFrame, expansion: pd.Series, change_window: int) -> pd.DataFrame:
    weights = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for i in range(change_window, len(panel), _REBALANCE_DAYS):
        target = pd.Series(0.0, index=panel.columns)
        chosen = EXPANSION_ASSETS if bool(expansion.iloc[i]) else CONTRACTION_ASSETS
        for asset in chosen:
            if asset in target.index:
                target[asset] = 1.0 / len(chosen)
        weights.iloc[i] = target.values
    return weights.shift(1)


def _segment(panel: pd.DataFrame, weights: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub_panel = panel.loc[mask]
    sub_weights = weights.loc[mask]
    portfolio = _run_vbt(sub_panel, sub_weights)
    benchmark = equal_weight_benchmark_return_pct(sub_panel)
    metrics = _metrics(portfolio, benchmark)

    checks = _robustness(metrics, sub_panel, sub_weights, start, end)
    result = BacktestResult(
        strategy_name="macro_regime",
        asset="MACRO[" + ",".join(ASSETS) + "]",
        start_date=start,
        end_date=end,
        engine="vectorbt_macro",
        metrics=metrics,
        robustness_results=checks,
    )
    report = build_research_report(
        user_request="macro regime rotation",
        strategy_spec=None,
        critique=StrategyCritique(approved=True),
        backtest_result=result,
        errors=[],
    )
    return {
        "verdict": report.verdict,
        "total_return_pct": metrics.total_return_pct,
        "benchmark_pct": metrics.buy_and_hold_return_pct,
        "sharpe": metrics.sharpe_ratio,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "failed_checks": [c.test_name for c in checks if not c.passed],
    }


def _run_vbt(panel: pd.DataFrame, weights: pd.DataFrame):
    vbt = _import_vectorbt()
    return vbt.Portfolio.from_orders(
        close=panel,
        size=weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=DEFAULT_INITIAL_CASH,
        fees=DEFAULT_COMMISSION_PCT,
        slippage=DEFAULT_SLIPPAGE_PCT,
        freq="1D",
    )


def _metrics(portfolio, benchmark_return_pct: float) -> BacktestMetrics:
    total = _safe_float(portfolio.total_return()) * 100
    return BacktestMetrics(
        total_return_pct=total,
        buy_and_hold_return_pct=benchmark_return_pct,
        sharpe_ratio=_safe_optional_float(portfolio.sharpe_ratio()),
        max_drawdown_pct=_safe_float(portfolio.max_drawdown()) * 100,
        num_trades=int(portfolio.trades.count()),
        win_rate_pct=None,
        exposure_time_pct=None,
        final_equity=_safe_float(portfolio.final_value()),
        beats_benchmark=total > benchmark_return_pct,
    )


def _robustness(metrics, panel, weights, start, end) -> list[RobustnessResult]:
    n_obs = estimate_trading_days(start, end)
    psr = (
        probabilistic_sharpe_ratio(metrics.sharpe_ratio, n_obs)
        if metrics.sharpe_ratio is not None
        else 0.0
    )
    return [
        RobustnessResult(
            test_name="Benchmark comparison",
            passed=metrics.total_return_pct > metrics.buy_and_hold_return_pct,
            details=f"{metrics.total_return_pct:.1f}% vs equal-weight {metrics.buy_and_hold_return_pct:.1f}%.",
        ),
        RobustnessResult(
            test_name="Positive return",
            passed=metrics.total_return_pct > 0,
            details=f"Total return {metrics.total_return_pct:.1f}%.",
        ),
        RobustnessResult(
            test_name="Drawdown sanity",
            passed=metrics.max_drawdown_pct > -50,
            details=f"Max drawdown {metrics.max_drawdown_pct:.1f}%.",
        ),
        RobustnessResult(
            test_name="Sharpe ratio significance (PSR)",
            passed=psr >= _PSR_PASS,
            details=f"PSR={psr:.3f} (Sharpe={metrics.sharpe_ratio}, n_obs~{n_obs}).",
        ),
        _walk_forward(panel, weights),
    ]


def _walk_forward(panel: pd.DataFrame, weights: pd.DataFrame) -> RobustnessResult:
    min_window = 252
    if len(panel) < min_window * 2:
        return RobustnessResult(
            test_name="Macro walk-forward stability",
            passed=False,
            details=f"Only {len(panel)} rows; need {min_window * 2} for walk-forward.",
        )
    n = min(4, len(panel) // min_window)
    returns: list[float] = []
    beats: list[bool] = []
    for win_panel, win_weights in zip(
        np.array_split(panel, n), np.array_split(weights, n), strict=False
    ):
        if len(win_panel) < min_window:
            continue
        pf = _run_vbt(win_panel, win_weights)
        ret = _safe_float(pf.total_return()) * 100
        returns.append(ret)
        beats.append(ret > equal_weight_benchmark_return_pct(win_panel))
    if not returns:
        return RobustnessResult(test_name="Macro walk-forward stability", passed=False, details="No windows.")
    positive = sum(r > 0 for r in returns)
    beat = sum(beats)
    passed = positive / len(returns) >= 0.5 and beat / len(returns) >= 0.5
    return RobustnessResult(
        test_name="Macro walk-forward stability",
        passed=passed,
        details=f"{positive}/{len(returns)} positive; {beat}/{len(returns)} beat equal-weight.",
    )


def format_macro_regime(result: MacroRegimeResult) -> str:
    lines = [
        f"Signal:  {result['signal']}  (pre-registered, frozen mapping)",
        f"Assets:  {', '.join(result['assets'])}",
        "Mapping: EXPANSION -> SPY,QQQ,GLD,USO equal-weight; CONTRACTION -> 100% TLT",
        "",
    ]

    def _block(label: str, seg: dict[str, Any]) -> list[str]:
        sharpe = seg.get("sharpe")
        sharpe_s = f"{sharpe:.2f}" if sharpe is not None else "n/a"
        failed = ", ".join(seg.get("failed_checks", [])) or "none"
        return [
            f"{label}:",
            f"  return {seg['total_return_pct']:.1f}%  vs equal-weight {seg['benchmark_pct']:.1f}%  "
            f"Sharpe {sharpe_s}  maxDD {seg['max_drawdown_pct']:.1f}%",
            f"  verdict: {seg['verdict']}   failed: {failed}",
        ]

    lines += _block("Full period", result["full"])
    if "lockbox" in result:
        split = result["lockbox_split"]
        lines.append("")
        lines.append(
            f"Train {split['original_start']}..{split['train_end']}  |  "
            f"held-out {split['lockbox_start']}..{split['original_end']}"
        )
        lines += _block("Train", result["train"])
        lines += _block("Held-out lockbox", result["lockbox"])
        lines.append("")
        lines.append(f"CONFIRMED OUT OF SAMPLE: {'YES' if result['confirmed'] else 'NO'}")
        lines.append("")
        lines.append(
            "Reminder: liquidity regimes are few (a handful of QE/QT episodes), so even "
            "a pass is weak evidence and counts as another shot against your budget."
        )
    return "\n".join(lines)
