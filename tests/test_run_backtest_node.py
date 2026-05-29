import pandas as pd

from trading_research_agent.nodes import run_backtest
from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.schemas.strategy import (
    MarketDataSource,
    StrategyFamily,
    StrategySpec,
)


def make_spec(asset: str = "BTC-USD") -> StrategySpec:
    return StrategySpec(
        name="BTC SMA",
        asset=asset,
        strategy_family=StrategyFamily.SMA_CROSSOVER,
        start_date="2020-01-01",
        end_date="2021-01-01",
        fast_window=50,
        slow_window=200,
        hypothesis="BTC may show persistent long-term trends.",
    )


def fake_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1000.0, 1001.0],
        }
    )


def fake_result(spec: StrategySpec) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset=spec.asset,
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="backtesting_py",
        metrics=BacktestMetrics(
            total_return_pct=1.0,
            buy_and_hold_return_pct=0.5,
            sharpe_ratio=1.0,
            max_drawdown_pct=-1.0,
            num_trades=1,
            win_rate_pct=100.0,
            exposure_time_pct=50.0,
            final_equity=10100.0,
            beats_benchmark=True,
        ),
    )


def test_btc_uses_coinmetrics_even_when_spec_defaults_to_yfinance(monkeypatch) -> None:
    calls: list[str] = []
    spec = make_spec("BTC-USD")

    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_coinmetrics_btc",
        lambda start, end: calls.append(f"{start}:{end}") or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_coingecko",
        lambda asset, start, end: (_ for _ in ()).throw(AssertionError("no coingecko")),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(AssertionError("no yfinance")),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert calls == ["2020-01-01:2021-01-01"]
    assert state["strategy_spec"].data_source == MarketDataSource.COINMETRICS
    assert state["backtest_result"].asset == "BTC-USD"


def test_yfinance_failure_falls_back_to_stooq(monkeypatch) -> None:
    calls: list[str] = []
    spec = make_spec("SPY")

    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(ValueError("yahoo failed")),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_stooq",
        lambda asset, start, end: calls.append(asset) or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert calls == ["SPY"]
    assert state["strategy_spec"].data_source == MarketDataSource.STOOQ
    assert state["backtest_result"].asset == "SPY"


def test_yfinance_spec_routes_through_tiingo_when_api_key_is_set(monkeypatch) -> None:
    calls: list[str] = []
    spec = make_spec("SPY")

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_tiingo",
        lambda asset, start, end: calls.append(asset) or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(AssertionError("no yfinance")),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert calls == ["SPY"]
    assert state["strategy_spec"].data_source == MarketDataSource.TIINGO


def test_yfinance_spec_falls_through_to_yfinance_when_tiingo_fails(monkeypatch) -> None:
    yfinance_calls: list[str] = []
    spec = make_spec("SPY")

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_tiingo",
        lambda asset, start, end: (_ for _ in ()).throw(ValueError("tiingo down")),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_yfinance",
        lambda asset, start, end: yfinance_calls.append(asset) or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert yfinance_calls == ["SPY"]
    assert state["strategy_spec"].data_source == MarketDataSource.YFINANCE


def test_explicit_tiingo_data_source_routes_through_tiingo(monkeypatch) -> None:
    calls: list[str] = []
    spec = make_spec("GLD").model_copy(update={"data_source": MarketDataSource.TIINGO})

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_tiingo",
        lambda asset, start, end: calls.append(asset) or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert calls == ["GLD"]
    assert state["strategy_spec"].data_source == MarketDataSource.TIINGO


def test_qqq_uses_fred_without_trying_yfinance(monkeypatch) -> None:
    calls: list[str] = []
    spec = make_spec("QQQ")

    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_fred_nasdaq",
        lambda asset, start, end: calls.append(asset) or fake_data(),
    )
    monkeypatch.setattr(
        run_backtest,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(AssertionError("no yfinance")),
    )
    monkeypatch.setattr(
        run_backtest,
        "get_backtest_backend",
        lambda engine: type(
            "Backend",
            (),
            {"run": lambda self, spec, data: fake_result(spec)},
        )(),
    )

    state = run_backtest.run_backtest_node(
        {"strategy_spec": spec, "critique": StrategyCritique(approved=True)}
    )

    assert calls == ["QQQ"]
    assert state["strategy_spec"].data_source == MarketDataSource.FRED
    assert state["backtest_result"].asset == "QQQ"
