from trading_research_agent.backtesting.backends.backtesting_py_backend import (
    BacktestingPyBackend,
)
from trading_research_agent.backtesting.backends.base import BacktestBackend
from trading_research_agent.backtesting.backends.vectorbt_backend import VectorbtBackend
from trading_research_agent.schemas.strategy import BacktestEngine


def get_backtest_backend(engine: BacktestEngine) -> BacktestBackend:
    if engine == BacktestEngine.BACKTESTING_PY:
        return BacktestingPyBackend()
    if engine == BacktestEngine.VECTORBT:
        return VectorbtBackend()
    raise NotImplementedError(f"Backtest engine not implemented: {engine}")
