from typing import Protocol

import pandas as pd

from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.strategy import StrategySpec


class BacktestBackend(Protocol):
    name: str

    def run(self, spec: StrategySpec, data: pd.DataFrame) -> BacktestResult:
        ...
