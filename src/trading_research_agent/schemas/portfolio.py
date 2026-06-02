from enum import Enum

from pydantic import BaseModel, Field, model_validator

from trading_research_agent.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE_PCT,
)
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
    # Time-series momentum with inverse-volatility sizing: assets with positive
    # trailing return are held, but low-volatility assets get larger weights than
    # high-volatility assets. If no asset has positive momentum, the book sits in
    # cash. This avoids raw momentum being dominated by BTC/oil-level volatility.
    VOLATILITY_SCALED_MOMENTUM = "volatility_scaled_momentum"
    # Crisis-hedge overlay: hold a core risk asset while it is above its own SMA;
    # when it breaks below, exit to cash and hold a capped slice of a volatility
    # hedge (e.g. a VIX ETF) that profits if the downturn accelerates.
    CRISIS_HEDGE = "crisis_hedge"
    # Long-horizon cross-sectional reversal (De Bondt-Thaler): hold the bottom_k
    # assets by trailing return measured over a long window that EXCLUDES the most
    # recent `skip_recent_days`. The skip-recent gap is what makes this orthogonal
    # to 6-12mo momentum rather than merely inverted momentum.
    CROSS_SECTIONAL_REVERSAL = "cross_sectional_reversal"


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

    initial_cash: float = Field(default=DEFAULT_INITIAL_CASH, gt=0)
    commission_pct: float = Field(default=DEFAULT_COMMISSION_PCT, ge=0)
    slippage_pct: float = Field(default=DEFAULT_SLIPPAGE_PCT, ge=0)

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
    skip_recent_days: int = Field(
        default=252,
        description=(
            "cross_sectional_reversal only: trading days at the recent end of the "
            "lookback window to EXCLUDE when scoring (the skip-recent gap). Must be "
            "in [0, lookback_days)."
        ),
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
        if self.lookback_days > 1260:
            raise ValueError("lookback_days must be <= 1260 (~5 years)")
        if self.rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if self.rebalance_days > 252:
            raise ValueError("rebalance_days must be <= 252")

        if self.portfolio_family in (
            PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
            PortfolioFamily.DUAL_MOMENTUM,
            PortfolioFamily.CROSS_SECTIONAL_REVERSAL,
        ):
            if self.top_k < 1:
                raise ValueError("top_k must be >= 1")
            if self.top_k > len(self.assets):
                raise ValueError("top_k cannot exceed the number of assets")

        if self.portfolio_family == PortfolioFamily.CROSS_SECTIONAL_REVERSAL:
            if self.skip_recent_days < 0:
                raise ValueError("skip_recent_days must be >= 0")
            if self.skip_recent_days >= self.lookback_days:
                raise ValueError("skip_recent_days must be < lookback_days")

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
