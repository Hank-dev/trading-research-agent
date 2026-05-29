import os

from trading_research_agent.backtesting.backends.factory import get_backtest_backend
from trading_research_agent.schemas.strategy import MarketDataSource
from trading_research_agent.tools.data_loader import (
    is_bitcoin_asset,
    is_nasdaq_proxy_asset,
    load_ohlcv_coinmetrics_btc,
    load_ohlcv_coingecko,
    load_ohlcv_fred_nasdaq,
    load_ohlcv_stooq,
    load_ohlcv_tiingo,
    load_ohlcv_yfinance,
)


def run_backtest_node(state: dict) -> dict:
    critique = state.get("critique")
    if critique is not None and not critique.approved:
        return {}

    spec = state.get("strategy_spec")
    if spec is None:
        return {"errors": _append_error(state, "Cannot run backtest without StrategySpec")}

    try:
        if spec.data_source == MarketDataSource.COINMETRICS or is_bitcoin_asset(spec.asset):
            spec = spec.model_copy(update={"data_source": MarketDataSource.COINMETRICS})
            data = load_ohlcv_coinmetrics_btc(spec.start_date, spec.end_date)
        elif spec.data_source == MarketDataSource.FRED or is_nasdaq_proxy_asset(spec.asset):
            spec = spec.model_copy(update={"data_source": MarketDataSource.FRED})
            data = load_ohlcv_fred_nasdaq(spec.asset, spec.start_date, spec.end_date)
        elif spec.data_source == MarketDataSource.COINGECKO:
            data = load_ohlcv_coingecko(spec.asset, spec.start_date, spec.end_date)
        elif spec.data_source == MarketDataSource.STOOQ:
            data = load_ohlcv_stooq(spec.asset, spec.start_date, spec.end_date)
        elif spec.data_source == MarketDataSource.TIINGO:
            data = load_ohlcv_tiingo(spec.asset, spec.start_date, spec.end_date)
        elif spec.data_source == MarketDataSource.YFINANCE:
            data, spec = _load_us_equity_with_fallbacks(spec)
        else:
            raise NotImplementedError(f"Data source not implemented: {spec.data_source}")
        backend = get_backtest_backend(spec.backtest_engine)
        return {"strategy_spec": spec, "backtest_result": backend.run(spec, data)}
    except Exception as exc:
        return {"errors": _append_error(state, f"Backtest failed: {exc}")}


def _append_error(state: dict, message: str) -> list[str]:
    return [*state.get("errors", []), message]


def _load_us_equity_with_fallbacks(spec):
    """For yfinance-spec'd assets, prefer Tiingo when TIINGO_API_KEY is set,
    fall through to yfinance, then to Stooq. Tiingo avoids the Yahoo bot-
    detection issues that plague yfinance on some systems."""
    if os.getenv("TIINGO_API_KEY"):
        try:
            data = load_ohlcv_tiingo(spec.asset, spec.start_date, spec.end_date)
            return data, spec.model_copy(update={"data_source": MarketDataSource.TIINGO})
        except ValueError:
            pass

    try:
        return load_ohlcv_yfinance(spec.asset, spec.start_date, spec.end_date), spec
    except ValueError as yfinance_error:
        try:
            data = load_ohlcv_stooq(spec.asset, spec.start_date, spec.end_date)
        except ValueError as stooq_error:
            raise ValueError(
                f"{yfinance_error} Stooq fallback also failed: {stooq_error}"
            ) from stooq_error
        return data, spec.model_copy(update={"data_source": MarketDataSource.STOOQ})
