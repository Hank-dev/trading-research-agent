"""Multi-asset portfolio research: per-spec pipeline + bounded pre-registered
exploration with a held-out lockbox.

Reuses the single-asset exploration helpers (winner selection, deflated-Sharpe
injection, lockbox split, failure summary) so the discipline is identical: a
pre-registered slate, no tweak-until-pass, a lockbox the slate never sees.
"""

from datetime import date
from typing import Any

import pandas as pd

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    PortfolioVectorbtBackend,
)
from trading_research_agent.nodes.generate_portfolio_slate import generate_portfolio_slate
from trading_research_agent.reports.markdown_report import build_research_report
from trading_research_agent.schemas.backtest import RobustnessResult
from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.schemas.portfolio import PortfolioSpec
from trading_research_agent.tools.data_loader import load_portfolio_panel
from trading_research_agent.tools.stats import (
    estimate_trading_days,
    probabilistic_sharpe_ratio,
)
from trading_research_agent.workflows.explore_research import (
    ExploreResult,
    _append_deflated_sharpe_checks,
    _select_winner,
    _summarize_failed_checks,
    _truncate_slate_for_lockbox,
)

_PSR_PASS_THRESHOLD = 0.95


def run_portfolio_backtest(
    spec: PortfolioSpec,
    user_request: str,
    *,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run one PortfolioSpec end to end: critique -> load -> backtest ->
    metric robustness -> report. Returns an explore-compatible state dict.

    If `panel` is provided (a pre-loaded master price panel), it is sliced to the
    spec's assets and date range instead of fetching from the network. This lets
    the robustness stress-tester run dozens of perturbations off one load.
    """
    critique = _portfolio_critique(spec)
    if not critique.approved:
        report = build_research_report(
            user_request=user_request,
            strategy_spec=spec,
            critique=critique,
            backtest_result=None,
            errors=[],
        )
        return {"strategy_spec": spec, "critique": critique, "report": report}

    try:
        data_panel = (
            _slice_panel(panel, spec)
            if panel is not None
            else load_portfolio_panel(spec.assets, spec.start_date, spec.end_date)
        )
        result = PortfolioVectorbtBackend().run(spec, data_panel)
    except Exception as exc:
        errors = [f"Portfolio backtest failed: {exc}"]
        report = build_research_report(
            user_request=user_request,
            strategy_spec=spec,
            critique=critique,
            backtest_result=None,
            errors=errors,
        )
        return {"strategy_spec": spec, "critique": critique, "report": report, "errors": errors}

    _prepend_metric_checks(result)
    report = build_research_report(
        user_request=user_request,
        strategy_spec=spec,
        critique=critique,
        backtest_result=result,
        errors=[],
    )
    return {
        "strategy_spec": spec,
        "critique": critique,
        "backtest_result": result,
        "report": report,
    }


def run_portfolio_exploration(
    user_request: str,
    slate_size: int,
    lockbox_pct: float = 0.0,
) -> ExploreResult:
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1")

    errors: list[str] = []
    try:
        slate = generate_portfolio_slate(user_request, slate_size)
    except Exception as exc:
        return {
            "candidates": [],
            "winner_index": None,
            "winner_reason": "slate_generation_failed",
            "errors": [f"Portfolio slate generation failed: {exc}"],
        }

    lockbox_split: dict[str, str] | None = None
    if lockbox_pct > 0:
        try:
            slate, lockbox_split = _truncate_slate_for_lockbox(slate, lockbox_pct)
        except Exception as exc:
            errors.append(f"Lockbox split failed: {exc}")
            lockbox_split = None

    candidates = [run_portfolio_backtest(spec, user_request) for spec in slate]
    _append_deflated_sharpe_checks(candidates)
    winner_index, winner_reason = _select_winner(candidates)

    result: ExploreResult = {
        "candidates": candidates,
        "winner_index": winner_index,
        "winner_reason": winner_reason,
        "failure_summary": _summarize_failed_checks(candidates),
    }
    if errors:
        result["errors"] = errors

    if lockbox_split is not None and winner_index is not None:
        winner_spec: PortfolioSpec = candidates[winner_index]["strategy_spec"]
        lockbox_spec = winner_spec.model_copy(
            update={
                "start_date": lockbox_split["lockbox_start"],
                "end_date": lockbox_split["original_end"],
                "name": f"{winner_spec.name} (lockbox)",
            }
        )
        lockbox_request = (
            f"{user_request}\n\nHeld-out lockbox re-test of the slate winner. "
            "This segment was not visible to slate generation."
        )
        result["lockbox"] = run_portfolio_backtest(lockbox_spec, lockbox_request)
        result["lockbox_split"] = lockbox_split

    return result


def run_portfolio_spec(
    spec: PortfolioSpec,
    user_request: str,
    lockbox_pct: float = 0.0,
) -> ExploreResult:
    """Run one hand-specified portfolio strategy deterministically.

    This intentionally skips slate generation and DSR. There is no cross-candidate
    search inside this workflow; the only optional extra gate is the held-out
    lockbox re-test of the specified rule.
    """
    errors: list[str] = []
    candidate_spec = spec
    lockbox_split: dict[str, str] | None = None

    if lockbox_pct > 0:
        try:
            truncated, lockbox_split = _truncate_slate_for_lockbox([spec], lockbox_pct)
            candidate_spec = truncated[0]
        except Exception as exc:
            errors.append(f"Lockbox split failed: {exc}")
            lockbox_split = None

    candidate = run_portfolio_backtest(candidate_spec, user_request)
    candidates = [candidate]
    winner_index = 0 if candidate.get("backtest_result") is not None else None

    result: ExploreResult = {
        "candidates": candidates,
        "winner_index": winner_index,
        "winner_reason": (
            "hand_specified_strategy"
            if winner_index is not None
            else "specified_strategy_produced_no_backtest"
        ),
        "failure_summary": _summarize_failed_checks(candidates),
    }
    if errors:
        result["errors"] = errors

    if lockbox_split is not None and winner_index is not None:
        lockbox_spec = candidate_spec.model_copy(
            update={
                "start_date": lockbox_split["lockbox_start"],
                "end_date": lockbox_split["original_end"],
                "name": f"{candidate_spec.name} (lockbox)",
            }
        )
        lockbox_request = (
            f"{user_request}\n\nHeld-out lockbox re-test of the specified portfolio. "
            "This segment was not visible to the in-sample evaluation."
        )
        result["lockbox"] = run_portfolio_backtest(lockbox_spec, lockbox_request)
        result["lockbox_split"] = lockbox_split

    return result


def _slice_panel(panel: pd.DataFrame, spec: PortfolioSpec, min_rows: int = 300) -> pd.DataFrame:
    cols = [a for a in spec.assets if a in panel.columns]
    if len(cols) < 2:
        raise ValueError(
            f"Pre-loaded panel is missing assets for {spec.assets}; "
            f"only {cols} available."
        )
    start = pd.Timestamp(spec.start_date)
    end = pd.Timestamp(spec.end_date)
    sub = panel[cols]
    sub = sub.loc[(sub.index >= start) & (sub.index <= end)].dropna(how="any")
    if len(sub) < min_rows:
        raise ValueError(
            f"Only {len(sub)} rows in sliced panel for {spec.start_date}..{spec.end_date}; "
            f"need at least {min_rows}."
        )
    return sub


def _portfolio_critique(spec: PortfolioSpec) -> StrategyCritique:
    problems: list[str] = []
    warnings: list[str] = []
    required_changes: list[str] = []

    try:
        start = date.fromisoformat(spec.start_date)
        end = date.fromisoformat(spec.end_date)
        if end <= start:
            problems.append("End date must be after start date.")
            required_changes.append("Choose an end date after the start date.")
        elif (end - start).days < 365 * 3:
            warnings.append("Backtest period is less than 3 years.")
    except ValueError as exc:
        problems.append(f"Invalid ISO date: {exc}")
        required_changes.append("Use YYYY-MM-DD dates.")

    if len(spec.assets) < 2:
        problems.append("A portfolio needs at least 2 assets.")
    warnings.append(f"Portfolio rotates across {len(spec.assets)} assets only.")

    return StrategyCritique(
        approved=not problems,
        problems=problems,
        warnings=warnings,
        required_changes=required_changes,
    )


def _prepend_metric_checks(result) -> None:
    metrics = result.metrics
    checks = [
        RobustnessResult(
            test_name="Benchmark comparison",
            passed=metrics.total_return_pct > metrics.buy_and_hold_return_pct,
            details=(
                f"Strategy return {metrics.total_return_pct:.2f}% vs equal-weight "
                f"{metrics.buy_and_hold_return_pct:.2f}%."
            ),
        ),
        RobustnessResult(
            test_name="Positive return",
            passed=metrics.total_return_pct > 0,
            details=f"Total return was {metrics.total_return_pct:.2f}%.",
        ),
        RobustnessResult(
            test_name="Drawdown sanity",
            passed=metrics.max_drawdown_pct > -50,
            details=f"Max drawdown was {metrics.max_drawdown_pct:.2f}%; required above -50%.",
        ),
        RobustnessResult(test_name="Sharpe ratio significance (PSR)", **_psr_check(result, metrics)),
    ]
    result.robustness_results = [*checks, *result.robustness_results]


def _psr_check(result, metrics) -> dict:
    if metrics.sharpe_ratio is None:
        return {"passed": False, "details": "Sharpe ratio was unavailable; cannot compute PSR."}
    n_obs = estimate_trading_days(result.start_date, result.end_date)
    psr = probabilistic_sharpe_ratio(metrics.sharpe_ratio, n_obs)
    return {
        "passed": psr >= _PSR_PASS_THRESHOLD,
        "details": (
            f"PSR={psr:.3f} (Sharpe={metrics.sharpe_ratio:.2f}, n_obs~{n_obs}); "
            f"required >= {_PSR_PASS_THRESHOLD:.2f} for pass."
        ),
    }
