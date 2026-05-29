from enum import Enum

from pydantic import BaseModel, Field, model_validator

from trading_research_agent.schemas.strategy import BacktestEngine, MarketDataSource


class PortfolioFamily(str, Enum):
    # Hold the top-K assets by trailing return, rebalanced periodically.
    CROSS_SECTIONAL_MOMENTUM = "cross_sectional_momentum"
    # Cross-sectional momentum, but only hold assets whose own trailing return is
    # positive; otherwise that allocation sits in cash (Antonacci-style defense).
    DUAL_MOMENTUM = "dual_momentum"
    # Equal-weight each universe asset, but only while it is above its own SMA.
    EQUAL_WEIGHT_TREND = "equal_weight_trend"
    # AQR-style time-series momentum: hold each asset (equal-weight slice) only
    # while its own trailing return is positive; the rest sits in cash. During a
    # broad downturn most assets switch off, so the book defensively de-risks
    # while any crisis-beneficiary (bonds, gold) keeps its slice.
    TIME_SERIES_MOMENTUM = "time_series_momentum"
    # Crisis-hedge overlay: hold a core risk asset while it is above its own SMA;
    # when it breaks below, exit to cash and hold a capped slice of a volatility
    # hedge (e.g. a VIX ETF) that profits if the downturn accelerates.
    CRISIS_HEDGE = "crisis_hedge"


class PortfolioSpec(BaseModel):
    name: str = Field(description="Short descriptive portfolio strategy name")
    assets: list[str] = Field(description="Universe of ticker symbols, >= 2")
    data_source: MarketDataSource | None = Field(
        default=None,
        description="Optional override; if unset, each asset is auto-routed by symbol.",
    )
    timeframe: str = Field(default="1d", description="MVP supports daily data only")
    portfolio_family: PortfolioFamily
    backtest_engine: BacktestEngine = BacktestEngine.VECTORBT

    start_date: str = Field(description="Backtest start date in YYYY-MM-DD format")
    end_date: str = Field(description="Backtest end date in YYYY-MM-DD format")

    initial_cash: float = Field(default=10_000, gt=0)
    commission_pct: float = Field(default=0.001, ge=0)
    slippage_pct: float = Field(default=0.0005, ge=0)

    benchmark: str = Field(default="equal_weight_buy_and_hold")

    lookback_days: int = Field(
        default=126,
        description="Trailing window for momentum / trend, in trading days.",
    )
    top_k: int = Field(
        default=1,
        description="How many assets to hold (momentum families).",
    )
    rebalance_days: int = Field(
        default=21,
        description="Rebalance frequency in trading days (21 ~ monthly).",
    )
    hedge_weight: float | None = Field(
        default=None,
        description=(
            "crisis_hedge only: fraction allocated to the volatility hedge during "
            "risk-off regimes, in (0, 0.5]. Vol ETFs are not portfolio-sized."
        ),
    )

    hypothesis: str = Field(
        description="Plain-English hypothesis explaining why this portfolio rule might work"
    )

    @model_validator(mode="after")
    def validate_portfolio(self) -> "PortfolioSpec":
        if self.timeframe != "1d":
            raise ValueError("MVP only supports daily timeframe: 1d")
        if self.backtest_engine != BacktestEngine.VECTORBT:
            raise ValueError("Portfolio strategies require the vectorbt engine")

        # Dedupe assets case-insensitively while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for raw in self.assets:
            asset = raw.strip()
            if asset and asset.lower() not in seen:
                seen.add(asset.lower())
                deduped.append(asset)
        object.__setattr__(self, "assets", deduped)

        if len(self.assets) < 2:
            raise ValueError("A portfolio needs at least 2 distinct assets")

        if self.lookback_days < 20:
            raise ValueError("lookback_days must be >= 20")
        if self.lookback_days > 504:
            raise ValueError("lookback_days must be <= 504 (~2 years)")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if self.rebalance_days > 252:
            raise ValueError("rebalance_days must be <= 252")

        if self.portfolio_family in (
            PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
            PortfolioFamily.DUAL_MOMENTUM,
        ):
            if self.top_k < 1:
                raise ValueError("top_k must be >= 1")
            if self.top_k > len(self.assets):
                raise ValueError("top_k cannot exceed the number of assets")

        if self.portfolio_family == PortfolioFamily.CRISIS_HEDGE:
            if len(self.assets) != 2:
                raise ValueError(
                    "crisis_hedge requires exactly 2 assets: [core_risk_asset, volatility_hedge]"
                )
            if self.hedge_weight is None:
                raise ValueError("crisis_hedge requires hedge_weight in (0, 0.5]")
            if not 0.0 < self.hedge_weight <= 0.5:
                raise ValueError("hedge_weight must be in (0, 0.5]")

        return self
