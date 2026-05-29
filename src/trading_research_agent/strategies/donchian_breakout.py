from backtesting import Strategy

from trading_research_agent.tools.indicators import donchian_high, donchian_low


class DonchianBreakoutStrategy(Strategy):
    entry_window = 55
    exit_window = 20

    def init(self) -> None:
        self.channel_high = self.I(donchian_high, self.data.High, self.entry_window)
        self.channel_low = self.I(donchian_low, self.data.Low, self.exit_window)

    def next(self) -> None:
        if not self.position and self.data.Close[-1] > self.channel_high[-1]:
            self.buy()
        elif self.position and self.data.Close[-1] < self.channel_low[-1]:
            self.position.close()
