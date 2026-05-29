from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("backtesting")

from trading_research_agent.backtesting.backends import backtesting_py_backend
from trading_research_agent.backtesting.backends.backtesting_py_backend import (
    BacktestingPyBackend,
)
from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.strategy import StrategyFamily, StrategySpec


def make_spec(**overrides) -> StrategySpec:
    data = {
        "name": "Test SMA",
        "asset": "SPY",
        "strategy_family": StrategyFamily.SMA_CROSSOVER,
        "start_date": "2020-01-01",
        "end_date": "2021-03-01",
        "fast_window": 5,
        "slow_window": 20,
        "commission_pct": 0.001,
        "slippage_pct": 0.0005,
        "hypothesis": "A short trend-following test may capture synthetic trends.",
    }
    data.update(overrides)
    return StrategySpec(**data)


def make_price_data(rows: int = 350) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = pd.Series(range(rows), index=index, dtype="float64") + 100
    close.iloc[100:180] = close.iloc[100:180].iloc[::-1].to_numpy()
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": [1000] * rows,
        },
        index=index,
    )


def test_backtesting_py_backend_runs_small_valid_dataframe(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = BacktestingPyBackend().run(make_spec(), make_price_data())

    assert isinstance(result, BacktestResult)
    assert result.engine == "backtesting_py"
    assert result.metrics.final_equity > 0


def test_unsupported_strategy_family_raises_clear_error() -> None:
    spec = SimpleNamespace(strategy_family="unsupported")

    with pytest.raises(ValueError, match="Unsupported strategy family"):
        BacktestingPyBackend()._select_strategy(spec)


def test_strategy_parameters_are_passed_correctly() -> None:
    params = BacktestingPyBackend()._strategy_params(make_spec(fast_window=7, slow_window=21))

    assert params == {"fast_window": 7, "slow_window": 21}


def test_effective_commission_includes_commission_plus_slippage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    class FakeBacktest:
        def __init__(
            self,
            data,
            strategy_cls,
            cash,
            commission,
            exclusive_orders,
            finalize_trades,
        ):
            captured["commission"] = commission

        def run(self, **kwargs):
            return pd.Series(
                {
                    "Return [%]": 5.0,
                    "Sharpe Ratio": 1.0,
                    "Max. Drawdown [%]": -3.0,
                    "# Trades": 22,
                    "Win Rate [%]": 55.0,
                    "Exposure Time [%]": 70.0,
                    "Equity Final [$]": 10500.0,
                    "_equity_curve": pd.DataFrame({"Equity": [10000.0, 10500.0]}),
                },
                dtype=object,
            )

    monkeypatch.setattr(backtesting_py_backend, "Backtest", FakeBacktest)
    monkeypatch.setattr(
        backtesting_py_backend,
        "save_equity_curve",
        lambda equity_curve, strategy_name: "outputs/test.png",
    )

    result = BacktestingPyBackend().run(make_spec(), make_price_data())

    assert captured["commission"] == pytest.approx(0.0015)
    assert result.equity_curve_path == "outputs/test.png"


def test_btc_prices_are_scaled_for_integer_backtest() -> None:
    backend = BacktestingPyBackend()
    data = make_price_data()
    scaled = backend._prepare_data_for_backtest(make_spec(asset="BTC-USD"), data)

    assert scaled["Close"].iloc[0] == pytest.approx(
        data["Close"].iloc[0] * backend.btc_fractional_unit
    )
    assert scaled["Volume"].iloc[0] == pytest.approx(
        data["Volume"].iloc[0] / backend.btc_fractional_unit
    )


def test_high_priced_non_btc_data_is_scaled_for_integer_backtest() -> None:
    backend = BacktestingPyBackend()
    data = make_price_data()
    data[["Open", "High", "Low", "Close"]] = data[["Open", "High", "Low", "Close"]] * 1000
    spec = make_spec(asset="QQQ", initial_cash=10_000)

    scaled = backend._prepare_data_for_backtest(spec, data)

    assert scaled["Close"].max() < spec.initial_cash
    assert scaled["Close"].iloc[0] < data["Close"].iloc[0]
