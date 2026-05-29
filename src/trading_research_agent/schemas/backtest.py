from pydantic import BaseModel, Field


class BacktestMetrics(BaseModel):
    total_return_pct: float
    buy_and_hold_return_pct: float
    sharpe_ratio: float | None
    max_drawdown_pct: float
    num_trades: int
    win_rate_pct: float | None
    exposure_time_pct: float | None
    final_equity: float
    beats_benchmark: bool


class RobustnessResult(BaseModel):
    test_name: str
    passed: bool
    details: str


class BacktestResult(BaseModel):
    strategy_name: str
    asset: str
    start_date: str
    end_date: str
    engine: str
    metrics: BacktestMetrics
    robustness_results: list[RobustnessResult] = Field(default_factory=list)
    equity_curve_path: str | None = None
