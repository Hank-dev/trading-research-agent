import pandas as pd
from backtesting import Backtest

from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec
from trading_research_agent.strategies.donchian_breakout import DonchianBreakoutStrategy
from trading_research_agent.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from trading_research_agent.strategies.sma_crossover import SmaCrossoverStrategy
from trading_research_agent.tools.metrics import (
    calculate_buy_and_hold_return,
    extract_backtesting_py_metrics,
)
from trading_research_agent.tools.data_loader import is_bitcoin_asset
from trading_research_agent.tools.plotting import save_equity_curve


class BacktestingPyBackend:
    name = "backtesting_py"
    btc_fractional_unit = 1 / 100_000_000

    def run(self, spec: StrategySpec, data: pd.DataFrame) -> BacktestResult:
        strategy_cls = self._select_strategy(spec)
        effective_commission = spec.commission_pct + spec.slippage_pct
        backtest_data = self._prepare_data_for_backtest(spec, data)

        bt = Backtest(
            backtest_data,
            strategy_cls,
            cash=spec.initial_cash,
            commission=effective_commission,
            exclusive_orders=True,
            finalize_trades=True,
        )
        stats = bt.run(**self._strategy_params(spec))
        buy_and_hold_return_pct = calculate_buy_and_hold_return(data)
        metrics = extract_backtesting_py_metrics(stats, buy_and_hold_return_pct)
        equity_curve_path = self._save_equity_curve_if_available(stats, spec.name)

        return BacktestResult(
            strategy_name=spec.name,
            asset=spec.asset,
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine=self.name,
            metrics=metrics,
            equity_curve_path=equity_curve_path,
        )

    def _select_strategy(self, spec: StrategySpec) -> type:
        if spec.strategy_family == StrategyFamily.SMA_CROSSOVER:
            return SmaCrossoverStrategy
        if spec.strategy_family == StrategyFamily.DONCHIAN_BREAKOUT:
            return DonchianBreakoutStrategy
        if spec.strategy_family == StrategyFamily.RSI_MEAN_REVERSION:
            return RsiMeanReversionStrategy
        raise ValueError(f"Unsupported strategy family: {spec.strategy_family}")

    def _strategy_params(self, spec: StrategySpec) -> dict[str, int | float]:
        if spec.strategy_family == StrategyFamily.SMA_CROSSOVER:
            return {
                "fast_window": spec.fast_window,
                "slow_window": spec.slow_window,
            }
        if spec.strategy_family == StrategyFamily.DONCHIAN_BREAKOUT:
            return {
                "entry_window": spec.entry_window,
                "exit_window": spec.exit_window,
            }
        if spec.strategy_family == StrategyFamily.RSI_MEAN_REVERSION:
            return {
                "rsi_window": spec.rsi_window,
                "oversold_threshold": spec.oversold_threshold,
                "exit_threshold": spec.exit_threshold,
            }
        raise ValueError(f"Unsupported strategy family: {spec.strategy_family}")

    def _prepare_data_for_backtest(
        self, spec: StrategySpec, data: pd.DataFrame
    ) -> pd.DataFrame:
        prepared = data.copy()
        price_scale = self._price_scale_for_backtest(spec, prepared)
        if price_scale != 1:
            for column in ["Open", "High", "Low", "Close"]:
                prepared[column] = prepared[column] * price_scale
            prepared["Volume"] = prepared["Volume"] / price_scale
        return prepared

    def _price_scale_for_backtest(self, spec: StrategySpec, data: pd.DataFrame) -> float:
        if is_bitcoin_asset(spec.asset):
            return self.btc_fractional_unit
        max_close = float(data["Close"].max())
        if max_close > spec.initial_cash:
            return spec.initial_cash / (max_close * 100)
        return 1

    def _save_equity_curve_if_available(
        self, stats: pd.Series, strategy_name: str
    ) -> str | None:
        equity_curve = stats.get("_equity_curve")
        if not isinstance(equity_curve, pd.DataFrame) or "Equity" not in equity_curve:
            return None
        return save_equity_curve(equity_curve["Equity"], strategy_name)
