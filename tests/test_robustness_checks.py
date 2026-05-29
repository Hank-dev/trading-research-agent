from trading_research_agent.nodes.robustness_checks import robustness_checks_node
from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult


def make_result(sharpe: float | None = 1.0, start: str = "2020-01-01", end: str = "2024-01-01") -> BacktestResult:
    return BacktestResult(
        strategy_name="X",
        asset="SPY",
        start_date=start,
        end_date=end,
        engine="vectorbt",
        metrics=BacktestMetrics(
            total_return_pct=20.0,
            buy_and_hold_return_pct=10.0,
            sharpe_ratio=sharpe,
            max_drawdown_pct=-12.0,
            num_trades=30,
            win_rate_pct=55.0,
            exposure_time_pct=60.0,
            final_equity=12000.0,
            beats_benchmark=True,
        ),
    )


def _find(result: BacktestResult, name: str):
    return next(check for check in result.robustness_results if check.test_name == name)


def test_psr_check_present_when_sharpe_available() -> None:
    state = {"backtest_result": make_result(sharpe=1.0)}
    out = robustness_checks_node(state)
    check = _find(out["backtest_result"], "Sharpe ratio significance (PSR)")
    assert "PSR=" in check.details


def test_psr_check_passes_for_high_sharpe_long_history() -> None:
    state = {"backtest_result": make_result(sharpe=2.0, start="2018-01-01", end="2024-01-01")}
    out = robustness_checks_node(state)
    assert _find(out["backtest_result"], "Sharpe ratio significance (PSR)").passed


def test_psr_check_fails_for_zero_sharpe() -> None:
    state = {"backtest_result": make_result(sharpe=0.0)}
    out = robustness_checks_node(state)
    assert not _find(out["backtest_result"], "Sharpe ratio significance (PSR)").passed


def test_psr_check_fails_when_sharpe_missing() -> None:
    state = {"backtest_result": make_result(sharpe=None)}
    out = robustness_checks_node(state)
    check = _find(out["backtest_result"], "Sharpe ratio significance (PSR)")
    assert not check.passed
    assert "unavailable" in check.details.lower()


def test_robustness_checks_node_noop_without_result() -> None:
    assert robustness_checks_node({}) == {}
