import os

import numpy as np
import pandas as pd

from trading_research_agent.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    RobustnessResult,
)
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.tools.portfolio_signals import (
    compute_target_weights,
    equal_weight_benchmark_return_pct,
)


class PortfolioVectorbtBackend:
    """Runs a multi-asset rotation portfolio as a single cash-shared, grouped
    vectorbt portfolio driven by a target-weight matrix."""

    name = "vectorbt_portfolio"
    monte_carlo_seed = 42
    monte_carlo_runs = 500

    def run(
        self,
        spec: PortfolioSpec,
        panel: pd.DataFrame,
        aux: pd.DataFrame | None = None,
    ) -> BacktestResult:
        weights = compute_target_weights(panel, spec, aux)
        portfolio = self._run_portfolio(spec, panel, weights)
        benchmark_return_pct = self._benchmark_return_pct(spec, panel)
        metrics = self._metrics_from_portfolio(portfolio, benchmark_return_pct)
        robustness_results = [
            self._rebalance_count_check(weights),
            self._walk_forward_check(spec, panel, aux),
            self._monte_carlo_check(portfolio.returns()),
        ]

        return BacktestResult(
            strategy_name=spec.name,
            asset="PORTFOLIO[" + ",".join(spec.assets) + "]",
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine=self.name,
            metrics=metrics,
            robustness_results=robustness_results,
        )

    def _benchmark_return_pct(self, spec: PortfolioSpec, panel: pd.DataFrame) -> float:
        # For a crisis hedge, the honest benchmark is buy-and-hold of the CORE
        # risk asset alone — NOT an equal-weight basket that includes the
        # perpetually-bleeding volatility ETF (which would flatter the strategy).
        if spec.portfolio_family == PortfolioFamily.CRISIS_HEDGE:
            core = panel[spec.assets[0]]
            return float((core.iloc[-1] / core.iloc[0] - 1.0) * 100.0)
        return equal_weight_benchmark_return_pct(panel)

    def _run_portfolio(self, spec: PortfolioSpec, panel: pd.DataFrame, weights: pd.DataFrame):
        vbt = _import_vectorbt()
        return vbt.Portfolio.from_orders(
            close=panel,
            size=weights,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            init_cash=spec.initial_cash,
            fees=spec.commission_pct,
            slippage=spec.slippage_pct,
            freq="1D",
        )

    def _metrics_from_portfolio(
        self, portfolio, benchmark_return_pct: float
    ) -> BacktestMetrics:
        total_return_pct = _safe_float(portfolio.total_return()) * 100
        sharpe_ratio = _safe_optional_float(portfolio.sharpe_ratio())
        max_drawdown_pct = _safe_float(portfolio.max_drawdown()) * 100
        num_trades = int(portfolio.trades.count())
        win_rate = _safe_optional_float(portfolio.trades.win_rate())
        exposure_time = _safe_optional_float(portfolio.gross_exposure().mean())

        return BacktestMetrics(
            total_return_pct=total_return_pct,
            buy_and_hold_return_pct=benchmark_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            num_trades=num_trades,
            win_rate_pct=win_rate * 100 if win_rate is not None else None,
            exposure_time_pct=exposure_time * 100 if exposure_time is not None else None,
            final_equity=_safe_float(portfolio.final_value()),
            beats_benchmark=total_return_pct > benchmark_return_pct,
        )

    def _rebalance_count_check(self, weights: pd.DataFrame) -> RobustnessResult:
        rebalances = int(weights.notna().any(axis=1).sum())
        return RobustnessResult(
            test_name="Portfolio rebalance count",
            passed=rebalances >= 6,
            details=f"{rebalances} rebalance event(s); require at least 6 for evidence.",
        )

    def _walk_forward_check(
        self, spec: PortfolioSpec, panel: pd.DataFrame, aux: pd.DataFrame | None = None
    ) -> RobustnessResult:
        min_window_rows = 252
        max_windows = 4
        if len(panel) < min_window_rows * 2:
            return RobustnessResult(
                test_name="Portfolio walk-forward stability",
                passed=False,
                details=(
                    f"Only {len(panel)} rows available; need at least "
                    f"{min_window_rows * 2} for walk-forward slices."
                ),
            )

        windows = np.array_split(panel, min(max_windows, len(panel) // min_window_rows))
        returns: list[float] = []
        benchmark_returns: list[float] = []
        for window in windows:
            if len(window) < spec.lookback_days + spec.rebalance_days:
                continue
            weights = compute_target_weights(window, spec, aux)
            portfolio = self._run_portfolio(spec, window, weights)
            returns.append(_safe_float(portfolio.total_return()) * 100)
            benchmark_returns.append(equal_weight_benchmark_return_pct(window))

        if not returns:
            return RobustnessResult(
                test_name="Portfolio walk-forward stability",
                passed=False,
                details="No valid walk-forward windows could be evaluated.",
            )

        positive_count = sum(value > 0 for value in returns)
        beat_count = sum(
            strategy > benchmark
            for strategy, benchmark in zip(returns, benchmark_returns, strict=False)
        )
        passed = positive_count / len(returns) >= 0.5 and beat_count / len(returns) >= 0.5
        return RobustnessResult(
            test_name="Portfolio walk-forward stability",
            passed=passed,
            details=(
                f"{positive_count}/{len(returns)} windows positive; "
                f"{beat_count}/{len(returns)} beat equal-weight; "
                f"returns={[round(value, 2) for value in returns]}."
            ),
        )

    def _monte_carlo_check(self, returns) -> RobustnessResult:
        clean_returns = returns.dropna().astype(float)
        if len(clean_returns) < 60:
            return RobustnessResult(
                test_name="Portfolio Monte Carlo return resampling",
                passed=False,
                details=f"Only {len(clean_returns)} daily returns available; need at least 60.",
            )

        rng = np.random.default_rng(self.monte_carlo_seed)
        values = clean_returns.to_numpy()
        samples = rng.choice(values, size=(self.monte_carlo_runs, len(values)), replace=True)
        equity = np.cumprod(1 + samples, axis=1)
        terminal_returns = equity[:, -1] - 1

        probability_positive = float(np.mean(terminal_returns > 0))
        fifth_pct_return = float(np.percentile(terminal_returns, 5) * 100)
        passed = probability_positive >= 0.6 and fifth_pct_return > -20

        return RobustnessResult(
            test_name="Portfolio Monte Carlo return resampling",
            passed=passed,
            details=(
                f"{self.monte_carlo_runs} seeded bootstrap paths; "
                f"probability positive={probability_positive:.2%}; "
                f"5th percentile return={fifth_pct_return:.2f}%."
            ),
        )


def _import_vectorbt():
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "Portfolio backend requires the 'vectorbt' package. Install with: "
            "python -m pip install -e .[dev]"
        ) from exc
    return vbt


def _safe_float(value: object) -> float:
    parsed = _safe_optional_float(value)
    if parsed is None:
        raise ValueError(f"Required VectorBT metric unavailable: {value!r}")
    return parsed


def _safe_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed) or np.isinf(parsed):
        return None
    return parsed
