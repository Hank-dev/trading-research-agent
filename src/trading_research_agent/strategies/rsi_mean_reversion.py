from backtesting import Strategy

from trading_research_agent.tools.indicators import rsi


class RsiMeanReversionStrategy(Strategy):
    rsi_window = 14
    oversold_threshold = 30
    exit_threshold = 50

    def init(self) -> None:
        self.rsi_values = self.I(rsi, self.data.Close, self.rsi_window)

    def next(self) -> None:
        if not self.position and self.rsi_values[-1] < self.oversold_threshold:
            self.buy()
        elif self.position and self.rsi_values[-1] > self.exit_threshold:
            self.position.close()
