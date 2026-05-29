from typing import TypedDict

from trading_research_agent.nodes.generate_slate import generate_slate
from trading_research_agent.schemas.backtest import RobustnessResult
from trading_research_agent.schemas.strategy import StrategySpec
from trading_research_agent.tools.dates import split_date_range
from trading_research_agent.tools.stats import (
    deflated_sharpe_ratio,
    estimate_trading_days,
)
from trading_research_agent.workflows.research_graph import ResearchState, build_research_graph


_DSR_PASS_THRESHOLD = 0.95


class FailureSummary(TypedDict):
    verdict_counts: dict[str, int]
    failed_check_counts: dict[str, int]
    candidates_with_backtest: int
    candidates_without_backtest: int


class ExploreResult(TypedDict, total=False):
    candidates: list[ResearchState]
    winner_index: int | None
    winner_reason: str
    lockbox: ResearchState
    lockbox_split: dict[str, str]
    failure_summary: FailureSummary
    errors: list[str]


def run_exploration(
    user_request: str,
    slate_size: int,
    lockbox_pct: float = 0.0,
) -> ExploreResult:
    """Generate a pre-registered slate, backtest every candidate, optionally
    re-run the winner on a held-out lockbox segment."""
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1")

    errors: list[str] = []

    try:
        slate = generate_slate(user_request, slate_size)
    except Exception as exc:
        return {
            "candidates": [],
            "winner_index": None,
            "winner_reason": "slate_generation_failed",
            "errors": [f"Slate generation failed: {exc}"],
        }

    lockbox_split: dict[str, str] | None = None
    if lockbox_pct > 0:
        try:
            slate, lockbox_split = _truncate_slate_for_lockbox(slate, lockbox_pct)
        except Exception as exc:
            errors.append(f"Lockbox split failed: {exc}")
            lockbox_split = None

    graph = build_research_graph()
    candidates: list[ResearchState] = []
    for spec in slate:
        candidate = graph.invoke(
            {
                "user_request": user_request,
                "strategy_spec": spec,
            }
        )
        candidates.append(candidate)

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
        winner_spec: StrategySpec = candidates[winner_index]["strategy_spec"]
        lockbox_spec = winner_spec.model_copy(
            update={
                "start_date": lockbox_split["lockbox_start"],
                "end_date": lockbox_split["original_end"],
                "name": f"{winner_spec.name} (lockbox)",
            }
        )
        lockbox_state = graph.invoke(
            {
                "user_request": (
                    f"{user_request}\n\nHeld-out lockbox re-test of the slate winner. "
                    "This segment was not visible to slate generation."
                ),
                "strategy_spec": lockbox_spec,
            }
        )
        result["lockbox"] = lockbox_state
        result["lockbox_split"] = lockbox_split

    return result


def _truncate_slate_for_lockbox(
    slate: list[StrategySpec], lockbox_pct: float
) -> tuple[list[StrategySpec], dict[str, str]]:
    first = slate[0]
    train_end, lockbox_start = split_date_range(
        first.start_date, first.end_date, lockbox_pct
    )
    lockbox_split = {
        "original_start": first.start_date,
        "original_end": first.end_date,
        "train_end": train_end,
        "lockbox_start": lockbox_start,
    }
    truncated = [spec.model_copy(update={"end_date": train_end}) for spec in slate]
    return truncated, lockbox_split


def _select_winner(candidates: list[ResearchState]) -> tuple[int | None, str]:
    """Pick the best candidate. Prefer benchmark-beaters with highest Sharpe.
    Fall back to highest total return when Sharpe is unavailable."""
    scored: list[tuple[int, float, bool, float]] = []
    for index, state in enumerate(candidates):
        result = state.get("backtest_result")
        if result is None:
            continue
        metrics = result.metrics
        scored.append(
            (
                index,
                metrics.sharpe_ratio if metrics.sharpe_ratio is not None else float("-inf"),
                metrics.beats_benchmark,
                metrics.total_return_pct,
            )
        )

    if not scored:
        return None, "no_candidate_produced_a_backtest"

    beaters = [s for s in scored if s[2]]
    if beaters:
        winner = max(beaters, key=lambda s: (s[1], s[3]))
        return winner[0], "highest_sharpe_among_benchmark_beaters"

    winner = max(scored, key=lambda s: (s[1], s[3]))
    return winner[0], "highest_sharpe_overall_no_benchmark_beaters"


def _append_deflated_sharpe_checks(candidates: list[ResearchState]) -> None:
    """Append a Deflated Sharpe Ratio check to each candidate's robustness results,
    using the cross-trial Sharpe distribution of the slate."""
    sharpes = [
        state["backtest_result"].metrics.sharpe_ratio
        for state in candidates
        if state.get("backtest_result") is not None
        and state["backtest_result"].metrics.sharpe_ratio is not None
    ]
    n_trials = len(sharpes)

    for state in candidates:
        result = state.get("backtest_result")
        if result is None:
            continue

        metrics = result.metrics
        check = _build_dsr_check(result, metrics, sharpes, n_trials)
        result.robustness_results = [*result.robustness_results, check]


def _build_dsr_check(
    result, metrics, sharpes: list[float], n_trials: int
) -> RobustnessResult:
    if metrics.sharpe_ratio is None:
        return RobustnessResult(
            test_name="Deflated Sharpe ratio (DSR)",
            passed=False,
            details="Sharpe ratio was unavailable; cannot compute DSR.",
        )
    if n_trials < 2:
        return RobustnessResult(
            test_name="Deflated Sharpe ratio (DSR)",
            passed=False,
            details=(
                f"Only {n_trials} trial(s) had a Sharpe ratio; "
                "DSR requires >= 2 trials."
            ),
        )

    n_obs = estimate_trading_days(result.start_date, result.end_date)
    dsr = deflated_sharpe_ratio(metrics.sharpe_ratio, n_obs, sharpes)
    return RobustnessResult(
        test_name="Deflated Sharpe ratio (DSR)",
        passed=dsr >= _DSR_PASS_THRESHOLD,
        details=(
            f"DSR={dsr:.3f} across {n_trials} slate trials "
            f"(Sharpe={metrics.sharpe_ratio:.2f}, n_obs~{n_obs}); "
            f"required >= {_DSR_PASS_THRESHOLD:.2f} for pass."
        ),
    )


def _summarize_failed_checks(candidates: list[ResearchState]) -> FailureSummary:
    """Tally verdicts and failed robustness checks across the slate so the user
    can see which gates are hardest to clear without inspecting each candidate."""
    verdict_counts: dict[str, int] = {}
    failed_check_counts: dict[str, int] = {}
    with_backtest = 0
    without_backtest = 0

    for state in candidates:
        report = state.get("report")
        if report is not None:
            verdict_counts[report.verdict] = verdict_counts.get(report.verdict, 0) + 1

        result = state.get("backtest_result")
        if result is None:
            without_backtest += 1
            continue
        with_backtest += 1
        for check in result.robustness_results:
            if not check.passed:
                failed_check_counts[check.test_name] = (
                    failed_check_counts.get(check.test_name, 0) + 1
                )

    sorted_fails = dict(
        sorted(failed_check_counts.items(), key=lambda item: (-item[1], item[0]))
    )

    return {
        "verdict_counts": verdict_counts,
        "failed_check_counts": sorted_fails,
        "candidates_with_backtest": with_backtest,
        "candidates_without_backtest": without_backtest,
    }
