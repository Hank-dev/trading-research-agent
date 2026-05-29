from enum import Enum

from pydantic import BaseModel, Field, model_validator

from trading_research_agent.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE_PCT,
)


class StrategyFamily(str, Enum):
    SMA_CROSSOVER = "sma_crossover"
    DONCHIAN_BREAKOUT = "donchian_breakout"
    RSI_MEAN_REVERSION = "rsi_mean_reversion"


class MarketDataSource(str, Enum):
    YFINANCE = "yfinance"
    COINGECKO = "coingecko"
    COINMETRICS = "coinmetrics"
    STOOQ = "stooq"
    FRED = "fred"
    TIINGO = "tiingo"


class BacktestEngine(str, Enum):
    BACKTESTING_PY = "backtesting_py"
    VECTORBT = "vectorbt"
    NAUTILUS = "nautilus"


class StrategySpec(BaseModel):
    name: str = Field(description="Short descriptive strategy name")
    asset: str = Field(description="Ticker symbol, e.g. BTC-USD, SPY, QQQ")
    data_source: MarketDataSource = MarketDataSource.YFINANCE
    timeframe: str = Field(default="1d", description="MVP supports daily data only")
    strategy_family: StrategyFamily
    backtest_engine: BacktestEngine = BacktestEngine.VECTORBT

    start_date: str = Field(description="Backtest start date in YYYY-MM-DD format")
    end_date: str = Field(description="Backtest end date in YYYY-MM-DD format")

    initial_cash: float = Field(default=DEFAULT_INITIAL_CASH, gt=0)
    commission_pct: float = Field(default=DEFAULT_COMMISSION_PCT, ge=0, description="0.001 = 0.1%")
    slippage_pct: float = Field(default=DEFAULT_SLIPPAGE_PCT, ge=0, description="0.0005 = 0.05%")

    benchmark: str = Field(default="buy_and_hold")

    fast_window: int | None = None
    slow_window: int | None = None

    entry_window: int | None = None
    exit_window: int | None = None

    rsi_window: int | None = None
    oversold_threshold: float | None = None
    exit_threshold: float | None = None

    hypothesis: str = Field(
        description="Plain-English hypothesis explaining why this strategy might work"
    )

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> "StrategySpec":
        if self.timeframe != "1d":
            raise ValueError("MVP only supports daily timeframe: 1d")

        if self.backtest_engine == BacktestEngine.NAUTILUS:
            raise ValueError("Nautilus backend is not implemented")

        if self.strategy_family == StrategyFamily.SMA_CROSSOVER:
            if self.fast_window is None or self.slow_window is None:
                raise ValueError("SMA crossover requires fast_window and slow_window")
            if self.fast_window <= 1:
                raise ValueError("fast_window must be > 1")
            if self.slow_window <= self.fast_window:
                raise ValueError("slow_window must be greater than fast_window")
            if self.slow_window > 300:
                raise ValueError("slow_window must be <= 300")

        if self.strategy_family == StrategyFamily.DONCHIAN_BREAKOUT:
            if self.entry_window is None or self.exit_window is None:
                raise ValueError("Donchian breakout requires entry_window and exit_window")
            if self.entry_window < 5:
                raise ValueError("entry_window must be >= 5")
            if self.exit_window < 2:
                raise ValueError("exit_window must be >= 2")
            if self.entry_window > 300 or self.exit_window > 300:
                raise ValueError("Donchian windows must be <= 300")

        if self.strategy_family == StrategyFamily.RSI_MEAN_REVERSION:
            if (
                self.rsi_window is None
                or self.oversold_threshold is None
                or self.exit_threshold is None
            ):
                raise ValueError(
                    "RSI mean reversion requires rsi_window, "
                    "oversold_threshold, and exit_threshold"
                )
            if self.rsi_window < 2:
                raise ValueError("rsi_window must be >= 2")
            if self.oversold_threshold >= self.exit_threshold:
                raise ValueError("oversold_threshold must be lower than exit_threshold")
            if self.oversold_threshold < 5:
                raise ValueError("oversold_threshold must be >= 5")
            if self.exit_threshold > 95:
                raise ValueError("exit_threshold must be <= 95")

        return self
