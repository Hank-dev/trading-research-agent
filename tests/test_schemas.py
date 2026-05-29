import pytest
from pydantic import ValidationError

from trading_research_agent.schemas.strategy import (
    BacktestEngine,
    StrategyFamily,
    StrategySpec,
)


def make_spec(**overrides) -> StrategySpec:
    data = {
        "name": "Test SMA",
        "asset": "SPY",
        "strategy_family": StrategyFamily.SMA_CROSSOVER,
        "start_date": "2018-01-01",
        "end_date": "2024-01-01",
        "fast_window": 50,
        "slow_window": 200,
        "hypothesis": "A simple trend-following rule may capture persistent trends.",
    }
    data.update(overrides)
    return StrategySpec(**data)


def test_invalid_sma_windows_fail() -> None:
    with pytest.raises(ValidationError):
        make_spec(fast_window=50, slow_window=20)


def test_invalid_donchian_windows_fail() -> None:
    with pytest.raises(ValidationError):
        make_spec(
            strategy_family=StrategyFamily.DONCHIAN_BREAKOUT,
            fast_window=None,
            slow_window=None,
            entry_window=4,
            exit_window=20,
        )


def test_invalid_rsi_thresholds_fail() -> None:
    with pytest.raises(ValidationError):
        make_spec(
            strategy_family=StrategyFamily.RSI_MEAN_REVERSION,
            fast_window=None,
            slow_window=None,
            rsi_window=14,
            oversold_threshold=55,
            exit_threshold=50,
        )


def test_non_daily_timeframe_fails() -> None:
    with pytest.raises(ValidationError):
        make_spec(timeframe="1h")


def test_non_mvp_backtest_engine_fails() -> None:
    with pytest.raises(ValidationError):
        make_spec(backtest_engine=BacktestEngine.NAUTILUS)


def test_vectorbt_backtest_engine_is_valid() -> None:
    spec = make_spec(backtest_engine=BacktestEngine.VECTORBT)

    assert spec.backtest_engine == BacktestEngine.VECTORBT


def test_missing_required_parameters_fail() -> None:
    with pytest.raises(ValidationError):
        make_spec(fast_window=None)
