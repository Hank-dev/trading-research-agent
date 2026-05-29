from trading_research_agent.nodes.refine_strategy import format_refinement_context
from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec


def test_format_refinement_context_includes_backtest_output() -> None:
    spec = StrategySpec(
        name="BTC SMA",
        asset="BTC-USD",
        strategy_family=StrategyFamily.SMA_CROSSOVER,
        start_date="2020-01-01",
        end_date="2024-01-01",
        fast_window=50,
        slow_window=200,
        hypothesis="BTC may trend over long windows.",
    )
    result = BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="backtesting_py",
        metrics=BacktestMetrics(
            total_return_pct=-5.0,
            buy_and_hold_return_pct=100.0,
            sharpe_ratio=None,
            max_drawdown_pct=-25.0,
            num_trades=4,
            win_rate_pct=None,
            exposure_time_pct=20.0,
            final_equity=9500.0,
            beats_benchmark=False,
        ),
    )

    context = format_refinement_context(
        {
            "user_request": "Try BTC SMA",
            "strategy_spec": spec,
            "backtest_result": result,
            "report": ResearchReport(
                markdown="# Report",
                verdict="reject",
                reasons=[],
                next_tests=[],
            ),
        }
    )

    assert "Previous BacktestResult" in context
    assert '"num_trades":4' in context.replace(" ", "")
    assert "Propose one revised StrategySpec" in context
