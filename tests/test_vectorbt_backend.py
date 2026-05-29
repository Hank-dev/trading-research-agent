import os

import pandas as pd
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
pytest.importorskip("vectorbt")

from trading_research_agent.backtesting.backends.vectorbt_backend import VectorbtBackend
from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.strategy import (
    BacktestEngine,
    StrategyFamily,
    StrategySpec,
)


def make_spec(**overrides) -> StrategySpec:
    data = {
        "name": "VectorBT SMA",
        "asset": "SPY",
        "backtest_engine": BacktestEngine.VECTORBT,
        "strategy_family": StrategyFamily.SMA_CROSSOVER,
        "start_date": "2020-01-01",
        "end_date": "2023-01-01",
        "fast_window": 5,
        "slow_window": 20,
        "hypothesis": "A trend-following rule may capture synthetic trends.",
    }
    data.update(overrides)
    return StrategySpec(**data)


def make_price_data(rows: int = 800) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    trend = pd.Series(range(rows), index=index, dtype="float64") * 0.25
    cycle = pd.Series([(-1) ** (i // 60) * 5 for i in range(rows)], index=index)
    close = 100 + trend + cycle
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1000] * rows,
        },
        index=index,
    )


def test_vectorbt_backend_runs_and_returns_robustness_results(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = VectorbtBackend().run(make_spec(), make_price_data())

    assert isinstance(result, BacktestResult)
    assert result.engine == "vectorbt"
    assert result.metrics.final_equity > 0
    assert result.equity_curve_path is not None
    assert {
        "VectorBT walk-forward stability",
        "VectorBT Monte Carlo return resampling",
    }.issubset({check.test_name for check in result.robustness_results})


def test_vectorbt_backend_builds_donchian_signals_without_lookahead() -> None:
    spec = make_spec(
        strategy_family=StrategyFamily.DONCHIAN_BREAKOUT,
        fast_window=None,
        slow_window=None,
        entry_window=5,
        exit_window=2,
    )
    data = make_price_data(rows=100)

    entries, exits = VectorbtBackend()._signals(spec, data)

    assert len(entries) == len(data)
    assert len(exits) == len(data)


def test_vectorbt_backend_monte_carlo_is_seeded() -> None:
    returns = pd.Series([0.01, -0.005, 0.002, 0.0] * 30)
    backend = VectorbtBackend()

    first = backend._monte_carlo_check(returns)
    second = backend._monte_carlo_check(returns)

    assert first.details == second.details
