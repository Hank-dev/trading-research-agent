from trading_research_agent.reports.markdown_report import build_research_report
from trading_research_agent.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    RobustnessResult,
)
from trading_research_agent.schemas.critique import StrategyCritique


def make_result(*, walk_forward_passed: bool, with_walk_forward: bool = True) -> BacktestResult:
    checks = [
        RobustnessResult(test_name="Benchmark comparison", passed=True, details="x"),
        RobustnessResult(test_name="Positive return", passed=True, details="x"),
        RobustnessResult(test_name="Drawdown sanity", passed=True, details="x"),
        RobustnessResult(test_name="Sharpe ratio significance (PSR)", passed=True, details="x"),
    ]
    if with_walk_forward:
        checks.append(
            RobustnessResult(
                test_name="Portfolio walk-forward stability",
                passed=walk_forward_passed,
                details="x",
            )
        )
    return BacktestResult(
        strategy_name="S",
        asset="SPY",
        start_date="2015-01-01",
        end_date="2023-01-01",
        engine="vectorbt",
        metrics=BacktestMetrics(
            total_return_pct=40.0,
            buy_and_hold_return_pct=20.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=-18.0,
            num_trades=40,
            win_rate_pct=55.0,
            exposure_time_pct=70.0,
            final_equity=14000.0,
            beats_benchmark=True,
        ),
        robustness_results=checks,
    )


def _verdict(result: BacktestResult) -> str:
    report = build_research_report(
        user_request="x",
        strategy_spec=None,
        critique=StrategyCritique(approved=True),
        backtest_result=result,
        errors=[],
    )
    return report.verdict


def test_strong_metrics_with_passing_walk_forward_is_worth_paper_trading() -> None:
    assert _verdict(make_result(walk_forward_passed=True)) == "worth_paper_trading"


def test_failing_walk_forward_caps_at_needs_more_testing() -> None:
    # Identical strong headline metrics, but walk-forward failed -> gated down.
    assert _verdict(make_result(walk_forward_passed=False)) == "needs_more_testing"


def test_absent_walk_forward_does_not_gate() -> None:
    # No walk-forward check present at all -> not treated as a failure.
    assert _verdict(make_result(walk_forward_passed=True, with_walk_forward=False)) == (
        "worth_paper_trading"
    )


def test_pipeline_errors_are_not_strategy_rejects() -> None:
    report = build_research_report(
        user_request="x",
        strategy_spec=None,
        critique=StrategyCritique(approved=True),
        backtest_result=None,
        errors=["data source failed"],
    )

    assert report.verdict == "error"
    assert "infrastructure or data errors" in report.reasons[0]
