from trading_research_agent.schemas.backtest import RobustnessResult
from trading_research_agent.tools.stats import (
    estimate_trading_days,
    probabilistic_sharpe_ratio,
)


_PSR_PASS_THRESHOLD = 0.95


def robustness_checks_node(state: dict) -> dict:
    result = state.get("backtest_result")
    if result is None:
        return {}

    metrics = result.metrics
    checks = [
        RobustnessResult(
            test_name="Minimum trade count",
            passed=metrics.num_trades >= 20,
            details=f"Trade count was {metrics.num_trades}; required at least 20.",
        ),
        RobustnessResult(
            test_name="Benchmark comparison",
            passed=metrics.total_return_pct > metrics.buy_and_hold_return_pct,
            details=(
                f"Strategy return {metrics.total_return_pct:.2f}% vs buy-and-hold "
                f"{metrics.buy_and_hold_return_pct:.2f}%."
            ),
        ),
        RobustnessResult(
            test_name="Drawdown sanity",
            passed=metrics.max_drawdown_pct > -50,
            details=f"Max drawdown was {metrics.max_drawdown_pct:.2f}%; required above -50%.",
        ),
        RobustnessResult(
            test_name="Positive return",
            passed=metrics.total_return_pct > 0,
            details=f"Total return was {metrics.total_return_pct:.2f}%.",
        ),
        RobustnessResult(
            test_name="Sharpe ratio significance (PSR)",
            **_psr_check(result, metrics),
        ),
    ]
    result.robustness_results = [*checks, *result.robustness_results]
    return {"backtest_result": result}


def _psr_check(result, metrics) -> dict:
    if metrics.sharpe_ratio is None:
        return {
            "passed": False,
            "details": "Sharpe ratio was unavailable; cannot compute PSR.",
        }
    n_obs = estimate_trading_days(result.start_date, result.end_date)
    psr = probabilistic_sharpe_ratio(metrics.sharpe_ratio, n_obs)
    return {
        "passed": psr >= _PSR_PASS_THRESHOLD,
        "details": (
            f"PSR={psr:.3f} (Sharpe={metrics.sharpe_ratio:.2f}, "
            f"n_obs~{n_obs}); required >= {_PSR_PASS_THRESHOLD:.2f} for pass."
        ),
    }
