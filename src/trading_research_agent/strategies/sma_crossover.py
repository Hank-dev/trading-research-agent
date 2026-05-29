from backtesting import Strategy
from backtesting.lib import crossover

from trading_research_agent.tools.indicators import sma


class SmaCrossoverStrategy(Strategy):
    fast_window = 50
    slow_window = 200

    def init(self) -> None:
        self.fast_sma = self.I(sma, self.data.Close, self.fast_window)
        self.slow_sma = self.I(sma, self.data.Close, self.slow_window)

    def next(self) -> None:
        if crossover(self.fast_sma, self.slow_sma):
            self.buy()
        elif crossover(self.slow_sma, self.fast_sma):
            self.position.close()
