from pathlib import Path

import pandas as pd
import pytest

from trading_research_agent.tools import data_loader
from trading_research_agent.tools.data_loader import (
    is_bitcoin_asset,
    is_nasdaq_proxy_asset,
    load_ohlcv_for_asset,
    load_ohlcv_coinmetrics_btc,
    load_ohlcv_coingecko,
    load_ohlcv_fred_nasdaq,
    load_ohlcv_stooq,
    load_ohlcv_tiingo,
    load_ohlcv_yfinance,
    tiingo_cache_status,
)


def make_ohlcv(rows: int = 301, descending: bool = False) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    if descending:
        index = index[::-1]
    return pd.DataFrame(
        {
            "Open": [100.0] * rows,
            "High": [101.0] * rows,
            "Low": [99.0] * rows,
            "Close": [100.5] * rows,
            "Volume": [1000] * rows,
        },
        index=index,
    )


def patch_download(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(data_loader.yf, "download", lambda *args, **kwargs: frame)


def test_missing_ohlc_columns_raise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_ohlcv().drop(columns=["Close"])
    patch_download(monkeypatch, frame)

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        load_ohlcv_yfinance("SPY", "2020-01-01", "2021-01-01")


def test_empty_download_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_download(monkeypatch, pd.DataFrame())

    with pytest.raises(ValueError, match="network/DNS/TLS"):
        load_ohlcv_yfinance("BTC-USD", "2020-01-01", "2021-01-01")


def test_too_few_rows_raise_error(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_download(monkeypatch, make_ohlcv(rows=299))

    with pytest.raises(ValueError, match="Fewer than 300"):
        load_ohlcv_yfinance("SPY", "2020-01-01", "2021-01-01")


def test_rows_with_missing_ohlc_values_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_ohlcv(rows=302)
    frame.iloc[0, frame.columns.get_loc("Open")] = None
    patch_download(monkeypatch, frame)

    loaded = load_ohlcv_yfinance("SPY", "2020-01-01", "2021-01-01")

    assert len(loaded) == 301
    assert loaded["Open"].isna().sum() == 0


def test_multiindex_columns_are_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_ohlcv()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    patch_download(monkeypatch, frame)

    loaded = load_ohlcv_yfinance("SPY", "2020-01-01", "2021-01-01")

    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_output_index_is_sorted_ascending(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_download(monkeypatch, make_ohlcv(descending=True))

    loaded = load_ohlcv_yfinance("SPY", "2020-01-01", "2021-01-01")

    assert loaded.index.is_monotonic_increasing


def test_coingecko_loader_builds_daily_btc_ohlcv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = []
    volumes = []
    index = pd.date_range("2020-01-01", periods=301, freq="D", tz="UTC")
    for day_number, day in enumerate(index):
        timestamp = int(day.timestamp() * 1000)
        prices.append([timestamp, 100.0 + day_number])
        volumes.append([timestamp, 1000.0 + day_number])

    monkeypatch.setattr(
        data_loader,
        "_fetch_coingecko_market_chart_range",
        lambda **kwargs: {"prices": prices, "total_volumes": volumes},
    )

    loaded = load_ohlcv_coingecko("BTC-USD", "2020-01-01", "2020-10-28")

    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 301
    assert loaded.iloc[0]["Open"] == 100.0
    assert loaded.iloc[-1]["Close"] == 400.0
    assert loaded.index.tz is None


def test_bitcoin_asset_detection() -> None:
    assert is_bitcoin_asset("BTC-USD") is True
    assert is_bitcoin_asset("bitcoin") is True
    assert is_bitcoin_asset("SPY") is False


def test_coinmetrics_loader_builds_daily_btc_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = []
    index = pd.date_range("2020-01-01", periods=301, freq="D", tz="UTC")
    for day_number, day in enumerate(index):
        prices.append({"t": int(day.timestamp()), "v": 100.0 + day_number})

    monkeypatch.setattr(
        data_loader,
        "fetch_btc_coinmetrics_price_usd",
        lambda start, base_url: prices,
    )

    loaded = load_ohlcv_coinmetrics_btc("2020-01-01", "2020-10-28")

    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 301
    assert loaded.iloc[0]["Open"] == 100.0
    assert loaded.iloc[-1]["Close"] == 400.0
    assert loaded.iloc[-1]["Volume"] == 0.0
    assert loaded.index.tz is None


def test_stooq_loader_builds_daily_ohlcv(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = ["Date,Open,High,Low,Close,Volume"]
    for date in pd.date_range("2020-01-01", periods=301, freq="D"):
        rows.append(f"{date:%Y-%m-%d},100,101,99,100.5,1000")

    seen_urls: list[str] = []

    def fake_fetch_text(url: str, source_name: str) -> str:
        seen_urls.append(url)
        return "\n".join(rows)

    monkeypatch.setattr(data_loader, "_fetch_text", fake_fetch_text)

    loaded = load_ohlcv_stooq("QQQ", "2020-01-01", "2020-10-28")

    assert "qqq.us" in seen_urls[0]
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 301
    assert loaded.index.is_monotonic_increasing


def test_fred_nasdaq_loader_builds_close_only_ohlcv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = ["observation_date,NASDAQ100"]
    for day_number, date in enumerate(pd.date_range("2020-01-01", periods=301, freq="D")):
        rows.append(f"{date:%Y-%m-%d},{1000 + day_number}")

    monkeypatch.setattr(
        data_loader,
        "_fetch_text",
        lambda url, source_name: "\n".join(rows),
    )

    loaded = load_ohlcv_fred_nasdaq("QQQ", "2020-01-01", "2020-10-28")

    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 301
    assert loaded.iloc[0]["Open"] == 1000
    assert loaded.iloc[-1]["Close"] == 1300
    assert loaded.iloc[-1]["Volume"] == 0.0


def test_nasdaq_proxy_asset_detection() -> None:
    assert is_nasdaq_proxy_asset("QQQ") is True
    assert is_nasdaq_proxy_asset("nasdaq") is True
    assert is_nasdaq_proxy_asset("SPY") is False


def test_tiingo_loader_builds_adjusted_ohlcv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rows = []
    for day_number, date in enumerate(pd.date_range("2020-01-01", periods=301, freq="D")):
        rows.append(
            {
                "date": date.strftime("%Y-%m-%dT00:00:00.000Z"),
                "open": 100 + day_number,
                "high": 101 + day_number,
                "low": 99 + day_number,
                "close": 100.5 + day_number,
                "volume": 1000 + day_number,
                "adjOpen": 200 + day_number,
                "adjHigh": 201 + day_number,
                "adjLow": 199 + day_number,
                "adjClose": 200.5 + day_number,
                "adjVolume": 2000 + day_number,
            }
        )

    seen_urls: list[str] = []
    seen_keys: list[str] = []

    def fake_fetch(url: str, api_key: str) -> list[dict]:
        seen_urls.append(url)
        seen_keys.append(api_key)
        return rows

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "_fetch_tiingo_json", fake_fetch)

    loaded = load_ohlcv_tiingo("GLD", "2020-01-01", "2020-10-28")

    assert "tiingo/daily/gld/prices" in seen_urls[0]
    assert "startDate=2020-01-01" in seen_urls[0]
    assert "endDate=2020-10-28" in seen_urls[0]
    assert seen_keys == ["test-token"]
    assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(loaded) == 301
    # Adjusted prices used, not raw
    assert loaded.iloc[0]["Open"] == 200
    assert loaded.iloc[-1]["Close"] == 500.5


def test_tiingo_loader_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="TIINGO_API_KEY"):
        load_ohlcv_tiingo("GLD", "2020-01-01", "2020-10-28")


def test_tiingo_loader_reuses_cached_covering_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rows = []
    for day_number, date in enumerate(pd.date_range("2020-01-01", periods=520, freq="D")):
        rows.append(
            {
                "date": date.strftime("%Y-%m-%dT00:00:00.000Z"),
                "adjOpen": 200 + day_number,
                "adjHigh": 201 + day_number,
                "adjLow": 199 + day_number,
                "adjClose": 200.5 + day_number,
                "adjVolume": 2000 + day_number,
            }
        )

    calls: list[str] = []

    def fake_fetch(url: str, api_key: str) -> list[dict]:
        calls.append(url)
        return rows

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "_fetch_tiingo_json", fake_fetch)

    first = load_ohlcv_tiingo("GLD", "2020-01-01", "2021-06-03")
    assert len(first) == 520

    monkeypatch.setattr(
        data_loader,
        "_fetch_tiingo_json",
        lambda url, api_key: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    second = load_ohlcv_tiingo("GLD", "2020-03-01", "2021-02-01")

    assert len(calls) == 1
    assert len(second) > 300
    assert second.index.min().date().isoformat() == "2020-03-01"
    assert second.index.max().date().isoformat() == "2021-02-01"


def test_tiingo_cache_status_reports_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rows = []
    for day_number, date in enumerate(pd.date_range("2020-01-01", periods=520, freq="D")):
        rows.append(
            {
                "date": date.strftime("%Y-%m-%dT00:00:00.000Z"),
                "adjOpen": 200 + day_number,
                "adjHigh": 201 + day_number,
                "adjLow": 199 + day_number,
                "adjClose": 200.5 + day_number,
                "adjVolume": 2000 + day_number,
            }
        )

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "_fetch_tiingo_json", lambda url, api_key: rows)

    load_ohlcv_tiingo("GLD", "2020-01-01", "2021-06-03")
    status = tiingo_cache_status("GLD", "2020-03-01", "2021-02-01")

    assert status["enabled"] is True
    assert status["covered"] is True
    assert status["rows"] > 300
    assert status["total_rows"] == 520


def test_auto_loader_names_tiingo_when_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setattr(
        data_loader,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(ValueError("yahoo tls")),
    )
    monkeypatch.setattr(
        data_loader,
        "load_ohlcv_stooq",
        lambda asset, start, end: (_ for _ in ()).throw(ValueError("stooq key")),
    )

    with pytest.raises(ValueError) as excinfo:
        load_ohlcv_for_asset("GLD", "2020-01-01", "2020-10-28")

    message = str(excinfo.value)
    assert "TIINGO_API_KEY is not set" in message
    assert "yfinance failed: yahoo tls" in message
    assert "Stooq fallback failed: stooq key" in message


def test_auto_loader_stops_on_tiingo_quota_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(
        data_loader,
        "load_ohlcv_tiingo",
        lambda asset, start, end: (_ for _ in ()).throw(
            ValueError("Tiingo request failed with HTTP 429: hourly request allocation")
        ),
    )
    monkeypatch.setattr(
        data_loader,
        "load_ohlcv_yfinance",
        lambda asset, start, end: (_ for _ in ()).throw(AssertionError("no yfinance")),
    )

    with pytest.raises(ValueError) as excinfo:
        load_ohlcv_for_asset("GLD", "2020-01-01", "2020-10-28")

    message = str(excinfo.value)
    assert "quota/auth" in message
    assert "HTTP 429" in message


def test_tiingo_loader_raises_on_empty_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "_fetch_tiingo_json", lambda url, api_key: [])
    with pytest.raises(ValueError, match="no daily rows"):
        load_ohlcv_tiingo("ZZZZ", "2020-01-01", "2020-10-28")


def test_tiingo_loader_raises_on_too_few_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rows = [
        {
            "date": "2020-01-0{}T00:00:00.000Z".format(i + 1),
            "adjOpen": 100,
            "adjHigh": 101,
            "adjLow": 99,
            "adjClose": 100.5,
            "adjVolume": 1000,
        }
        for i in range(5)
    ]
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setenv("TRADING_RESEARCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data_loader, "_fetch_tiingo_json", lambda url, api_key: rows)
    with pytest.raises(ValueError, match="Fewer than 300"):
        load_ohlcv_tiingo("GLD", "2020-01-01", "2020-01-05")
