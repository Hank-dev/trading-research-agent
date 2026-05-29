import os

import numpy as np
import pandas as pd

from trading_research_agent.schemas.backtest import (
    BacktestMetrics,
    BacktestResult,
    RobustnessResult,
)
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.tools.indicators import (
    donchian_high,
    donchian_low,
    rsi,
    sma,
)
from trading_research_agent.tools.metrics import calculate_buy_and_hold_return
from trading_research_agent.tools.plotting import save_equity_curve


class VectorbtBackend:
    name = "vectorbt"
    monte_carlo_seed = 42
    monte_carlo_runs = 500

    def run(self, spec: StrategySpec, data: pd.DataFrame) -> BacktestResult:
        portfolio, entries, exits = self._run_portfolio(spec, data)
        buy_and_hold_return_pct = calculate_buy_and_hold_return(data)
        metrics = self._metrics_from_portfolio(portfolio, buy_and_hold_return_pct)
        equity_curve_path = save_equity_curve(portfolio.value(), spec.name)
        robustness_results = [
            self._walk_forward_check(spec, data),
            self._monte_carlo_check(portfolio.returns()),
        ]

        return BacktestResult(
            strategy_name=spec.name,
            asset=spec.asset,
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine=self.name,
            metrics=metrics,
            robustness_results=robustness_results,
            equity_curve_path=equity_curve_path,
        )

    def _run_portfolio(self, spec: StrategySpec, data: pd.DataFrame):
        vbt = _import_vectorbt()
        close = data["Close"].astype(float)
        entries, exits = self._signals(spec, data)
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=spec.initial_cash,
            fees=spec.commission_pct,
            slippage=spec.slippage_pct,
            direction="longonly",
            freq="1D",
        )
        return portfolio, entries, exits

    def _signals(
        self, spec: StrategySpec, data: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        close = data["Close"].astype(float)
        if spec.strategy_family == StrategyFamily.SMA_CROSSOVER:
            fast = sma(close, spec.fast_window)
            slow = sma(close, spec.slow_window)
            entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
            return entries.fillna(False), exits.fillna(False)

        if spec.strategy_family == StrategyFamily.DONCHIAN_BREAKOUT:
            channel_high = donchian_high(data["High"].astype(float), spec.entry_window)
            channel_low = donchian_low(data["Low"].astype(float), spec.exit_window)
            entries = close > channel_high
            exits = close < channel_low
            return entries.fillna(False), exits.fillna(False)

        if spec.strategy_family == StrategyFamily.RSI_MEAN_REVERSION:
            rsi_values = rsi(close, spec.rsi_window)
            entries = rsi_values < spec.oversold_threshold
            exits = rsi_values > spec.exit_threshold
            return entries.fillna(False), exits.fillna(False)

        raise ValueError(f"Unsupported strategy family: {spec.strategy_family}")

    def _metrics_from_portfolio(
        self, portfolio, buy_and_hold_return_pct: float
    ) -> BacktestMetrics:
        total_return_pct = _safe_float(portfolio.total_return()) * 100
        sharpe_ratio = _safe_optional_float(portfolio.sharpe_ratio())
        max_drawdown_pct = _safe_float(portfolio.max_drawdown()) * 100
        num_trades = int(portfolio.trades.count())
        win_rate = _safe_optional_float(portfolio.trades.win_rate())
        exposure_time = _safe_optional_float(portfolio.gross_exposure().mean())

        return BacktestMetrics(
            total_return_pct=total_return_pct,
            buy_and_hold_return_pct=buy_and_hold_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            num_trades=num_trades,
            win_rate_pct=win_rate * 100 if win_rate is not None else None,
            exposure_time_pct=exposure_time * 100 if exposure_time is not None else None,
            final_equity=_safe_float(portfolio.final_value()),
            beats_benchmark=total_return_pct > buy_and_hold_return_pct,
        )

    def _walk_forward_check(self, spec: StrategySpec, data: pd.DataFrame) -> RobustnessResult:
        min_window_rows = 252
        max_windows = 4
        if len(data) < min_window_rows * 2:
            return RobustnessResult(
                test_name="VectorBT walk-forward stability",
                passed=False,
                details=(
                    f"Only {len(data)} rows available; need at least "
                    f"{min_window_rows * 2} for walk-forward slices."
                ),
            )

        windows = np.array_split(data, min(max_windows, len(data) // min_window_rows))
        returns: list[float] = []
        benchmark_returns: list[float] = []
        for window in windows:
            if len(window) < min_window_rows:
                continue
            portfolio, _, _ = self._run_portfolio(spec, window)
            returns.append(_safe_float(portfolio.total_return()) * 100)
            benchmark_returns.append(calculate_buy_and_hold_return(window))

        if not returns:
            return RobustnessResult(
                test_name="VectorBT walk-forward stability",
                passed=False,
                details="No valid walk-forward windows could be evaluated.",
            )

        positive_count = sum(value > 0 for value in returns)
        beat_count = sum(
            strategy > benchmark
            for strategy, benchmark in zip(returns, benchmark_returns, strict=False)
        )
        pass_rate = positive_count / len(returns)
        beat_rate = beat_count / len(returns)
        passed = pass_rate >= 0.5 and beat_rate >= 0.5
        return RobustnessResult(
            test_name="VectorBT walk-forward stability",
            passed=passed,
            details=(
                f"{positive_count}/{len(returns)} windows positive; "
                f"{beat_count}/{len(returns)} beat buy-and-hold; "
                f"returns={[round(value, 2) for value in returns]}."
            ),
        )

    def _monte_carlo_check(self, returns: pd.Series) -> RobustnessResult:
        clean_returns = returns.dropna().astype(float)
        if len(clean_returns) < 60:
            return RobustnessResult(
                test_name="VectorBT Monte Carlo return resampling",
                passed=False,
                details=f"Only {len(clean_returns)} daily returns available; need at least 60.",
            )

        rng = np.random.default_rng(self.monte_carlo_seed)
        values = clean_returns.to_numpy()
        samples = rng.choice(
            values,
            size=(self.monte_carlo_runs, len(values)),
            replace=True,
        )
        equity = np.cumprod(1 + samples, axis=1)
        terminal_returns = equity[:, -1] - 1
        running_max = np.maximum.accumulate(equity, axis=1)
        max_drawdowns = np.min(equity / running_max - 1, axis=1)

        probability_positive = float(np.mean(terminal_returns > 0))
        fifth_pct_return = float(np.percentile(terminal_returns, 5) * 100)
        fifth_pct_drawdown = float(np.percentile(max_drawdowns, 5) * 100)
        passed = probability_positive >= 0.6 and fifth_pct_return > -20

        return RobustnessResult(
            test_name="VectorBT Monte Carlo return resampling",
            passed=passed,
            details=(
                f"{self.monte_carlo_runs} seeded bootstrap paths; "
                f"probability positive={probability_positive:.2%}; "
                f"5th percentile return={fifth_pct_return:.2f}%; "
                f"5th percentile max drawdown={fifth_pct_drawdown:.2f}%."
            ),
        )


def _import_vectorbt():
    # VectorBT 1.0.0 can fail import in this Python 3.14 environment when Numba
    # tries to cache package functions. Disabling JIT keeps this MVP backend
    # usable and deterministic; it can be revisited under Python 3.11/3.12.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "VectorBT backend requires the 'vectorbt' package. Install project "
            "dependencies with: python -m pip install -e .[dev]"
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
