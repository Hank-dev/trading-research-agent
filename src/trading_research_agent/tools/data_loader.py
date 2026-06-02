from datetime import UTC, datetime, time, timedelta
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from io import StringIO
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
COINGECKO_PUBLIC_BASE_URL = "https://api.coingecko.com/api/v3"
COINMETRICS_COMMUNITY_BASE_URL = "https://community-api.coinmetrics.io/v4"
STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
FRED_GRAPH_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
DEFAULT_PRICE_CACHE_DIR = Path("outputs/cache")


def load_ohlcv_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    if data.empty:
        raise ValueError(
            f"No OHLCV data returned for {ticker}. yfinance returned an empty "
            "DataFrame. Common causes are Yahoo Finance network/DNS/TLS access "
            "failures, Yahoo throttling/consent issues, an invalid ticker, or a "
            "date range with no available data."
        )

    data = _normalize_yfinance_columns(data, ticker)
    _require_columns(data, REQUIRED_OHLCV_COLUMNS)
    data = _use_adjusted_prices_if_available(data)

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if len(data) < 300:
        raise ValueError(
            f"Fewer than 300 valid OHLC rows returned for {ticker}: {len(data)}"
        )

    return data[REQUIRED_OHLCV_COLUMNS]


def load_ohlcv_tiingo(
    ticker: str,
    start: str,
    end: str,
    base_url: str = TIINGO_BASE_URL,
) -> pd.DataFrame:
    """Load daily adjusted OHLCV from Tiingo's IEX/EOD endpoint. Requires TIINGO_API_KEY."""
    symbol = ticker.strip().lower()
    rows = _load_tiingo_rows(symbol, start, end, base_url=base_url)

    if not rows:
        raise ValueError(
            f"Tiingo returned no daily rows for {ticker} between {start} and {end}."
        )

    return _tiingo_rows_to_ohlcv(rows, ticker)


def _load_tiingo_rows(
    symbol: str,
    start: str,
    end: str,
    *,
    base_url: str,
) -> list[dict]:
    cache_path = _tiingo_cache_path(symbol, base_url)
    if _price_cache_enabled():
        cached = _read_tiingo_cache(cache_path)
        if _tiingo_cache_covers(cached, start, end):
            return _slice_tiingo_rows(cached.get("rows", []), start, end)

    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise ValueError(
            "TIINGO_API_KEY is not set. Add it to your .env file to use Tiingo."
        )

    query = urlencode({"startDate": start, "endDate": end, "format": "json"})
    url = f"{base_url.rstrip('/')}/{symbol}/prices?{query}"
    rows = _fetch_tiingo_json(url, api_key=api_key)
    if _price_cache_enabled() and rows:
        _write_tiingo_cache(cache_path, symbol, base_url, start, end, rows)
    return rows


def _tiingo_rows_to_ohlcv(rows: list[dict], ticker: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    expected_columns = ["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
    _require_columns(frame, expected_columns)

    frame["Date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame.set_index("Date").sort_index()
    frame = frame.dropna(subset=["adjOpen", "adjHigh", "adjLow", "adjClose"])

    daily = pd.DataFrame(index=frame.index)
    daily["Open"] = frame["adjOpen"].astype(float)
    daily["High"] = frame["adjHigh"].astype(float)
    daily["Low"] = frame["adjLow"].astype(float)
    daily["Close"] = frame["adjClose"].astype(float)
    daily["Volume"] = frame["adjVolume"].fillna(0).astype(float)

    if len(daily) < 300:
        raise ValueError(
            f"Fewer than 300 valid Tiingo OHLC rows returned for {ticker}: {len(daily)}"
        )

    return daily[REQUIRED_OHLCV_COLUMNS]


def _price_cache_enabled() -> bool:
    disabled = os.getenv("TRADING_RESEARCH_DISABLE_CACHE", "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def _price_cache_root() -> Path:
    configured = os.getenv("TRADING_RESEARCH_CACHE_DIR")
    return Path(configured) if configured else DEFAULT_PRICE_CACHE_DIR


def _tiingo_cache_path(symbol: str, base_url: str) -> Path:
    safe_symbol = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in symbol.lower()
    )
    source_hash = hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()[:8]
    return _price_cache_root() / "tiingo" / f"{safe_symbol}_{source_hash}.json"


def tiingo_cache_status(
    ticker: str,
    start: str,
    end: str,
    base_url: str = TIINGO_BASE_URL,
) -> dict[str, object]:
    """Inspect local Tiingo cache coverage without making a network request."""
    symbol = ticker.strip().lower()
    path = _tiingo_cache_path(symbol, base_url)
    if not _price_cache_enabled():
        return {
            "enabled": False,
            "path": str(path),
            "covered": False,
            "rows": 0,
            "total_rows": 0,
            "ranges": [],
        }

    payload = _read_tiingo_cache(path)
    rows = payload.get("rows", [])
    covered = _tiingo_cache_covers(payload, start, end)
    return {
        "enabled": True,
        "path": str(path),
        "covered": covered,
        "rows": len(_slice_tiingo_rows(rows, start, end)) if covered else 0,
        "total_rows": len(rows),
        "ranges": payload.get("ranges", []),
    }


def _read_tiingo_cache(path: Path) -> dict:
    if not path.exists():
        return {"ranges": [], "rows": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ranges": [], "rows": []}
    if not isinstance(payload, dict):
        return {"ranges": [], "rows": []}
    if not isinstance(payload.get("ranges"), list) or not isinstance(payload.get("rows"), list):
        return {"ranges": [], "rows": []}
    return payload


def _tiingo_cache_covers(cache: dict, start: str, end: str) -> bool:
    for item in cache.get("ranges", []):
        if not isinstance(item, dict):
            continue
        cached_start = item.get("start")
        cached_end = item.get("end")
        if isinstance(cached_start, str) and isinstance(cached_end, str):
            if cached_start <= start and cached_end >= end:
                return True
    return False


def _slice_tiingo_rows(rows: list[dict], start: str, end: str) -> list[dict]:
    sliced: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = _tiingo_row_day(row)
        if day is not None and start <= day <= end:
            sliced.append(row)
    return sorted(sliced, key=lambda row: _tiingo_row_day(row) or "")


def _write_tiingo_cache(
    path: Path,
    symbol: str,
    base_url: str,
    start: str,
    end: str,
    rows: list[dict],
) -> None:
    existing = _read_tiingo_cache(path)
    merged_rows = _merge_tiingo_rows(existing.get("rows", []), rows)
    ranges = _merge_tiingo_ranges(
        [
            item
            for item in existing.get("ranges", [])
            if isinstance(item, dict)
        ]
        + [{"start": start, "end": end}]
    )
    payload = {
        "source": "tiingo",
        "symbol": symbol,
        "base_url": base_url.rstrip("/"),
        "ranges": ranges,
        "rows": merged_rows,
        "updated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        pass


def _merge_tiingo_rows(existing: list, incoming: list[dict]) -> list[dict]:
    by_day: dict[str, dict] = {}
    for row in [*existing, *incoming]:
        if not isinstance(row, dict):
            continue
        day = _tiingo_row_day(row)
        if day is not None:
            by_day[day] = row
    return [by_day[day] for day in sorted(by_day)]


def _merge_tiingo_ranges(ranges: list[dict]) -> list[dict[str, str]]:
    parsed: list[tuple[datetime, datetime]] = []
    for item in ranges:
        raw_start = item.get("start")
        raw_end = item.get("end")
        if not isinstance(raw_start, str) or not isinstance(raw_end, str):
            continue
        try:
            start_dt = datetime.fromisoformat(raw_start)
            end_dt = datetime.fromisoformat(raw_end)
        except ValueError:
            continue
        if end_dt < start_dt:
            continue
        parsed.append((start_dt, end_dt))

    merged: list[list[datetime]] = []
    for start_dt, end_dt in sorted(parsed):
        if not merged or start_dt > merged[-1][1] + timedelta(days=1):
            merged.append([start_dt, end_dt])
        else:
            merged[-1][1] = max(merged[-1][1], end_dt)
    return [
        {"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()}
        for start_dt, end_dt in merged
    ]


def _tiingo_row_day(row: dict) -> str | None:
    raw = row.get("date")
    if raw is None:
        return None
    try:
        return pd.Timestamp(raw).date().isoformat()
    except (TypeError, ValueError):
        return None


def _fetch_tiingo_json(url: str, api_key: str) -> list[dict]:
    request = Request(
        url,
        headers={
            "accept": "application/json",
            "Authorization": f"Token {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Tiingo request failed with HTTP {exc.code}: {body[:300]}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"Tiingo request failed: {exc}") from exc

    if isinstance(payload, dict) and "detail" in payload:
        raise ValueError(f"Tiingo error: {payload['detail']}")
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Tiingo response shape: {type(payload).__name__}")
    return payload


def load_ohlcv_stooq(
    ticker: str,
    start: str,
    end: str,
    base_url: str = STOOQ_BASE_URL,
) -> pd.DataFrame:
    symbol = _stooq_symbol(ticker)
    query_params = {
        "s": symbol,
        "d1": start.replace("-", ""),
        "d2": end.replace("-", ""),
        "i": "d",
    }
    if os.getenv("STOOQ_API_KEY"):
        query_params["apikey"] = os.environ["STOOQ_API_KEY"]
    query = urlencode(query_params)
    csv_text = _fetch_text(f"{base_url}?{query}", source_name="Stooq")
    if "Get your apikey" in csv_text:
        raise ValueError(
            "Stooq now requires an API key for CSV downloads. Set STOOQ_API_KEY "
            "or use another data source."
        )
    data = pd.read_csv(StringIO(csv_text))
    if data.empty:
        raise ValueError(f"No OHLCV data returned from Stooq for {ticker}")

    _require_columns(data, REQUIRED_OHLCV_COLUMNS + ["Date"])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date").sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if len(data) < 300:
        raise ValueError(
            f"Fewer than 300 valid Stooq OHLC rows returned for {ticker}: {len(data)}"
        )

    return data[REQUIRED_OHLCV_COLUMNS]


def load_ohlcv_fred_nasdaq(
    asset: str,
    start: str,
    end: str,
    base_url: str = FRED_GRAPH_BASE_URL,
) -> pd.DataFrame:
    return _load_ohlcv_fred_close_only(
        series_id=_fred_nasdaq_series_id(asset),
        asset_label="Nasdaq",
        start=start,
        end=end,
        base_url=base_url,
    )


def _load_ohlcv_fred_close_only(
    series_id: str,
    asset_label: str,
    start: str,
    end: str,
    base_url: str,
) -> pd.DataFrame:
    query = urlencode({"id": series_id})
    csv_text = _fetch_text(f"{base_url}?{query}", source_name="FRED")
    data = pd.read_csv(StringIO(csv_text))
    expected_columns = ["observation_date", series_id]
    _require_columns(data, expected_columns)

    data["Date"] = pd.to_datetime(data["observation_date"])
    data["Close"] = pd.to_numeric(data[series_id].replace(".", pd.NA), errors="coerce")
    data = data.dropna(subset=["Close"]).set_index("Date").sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    data = data[(data.index >= start_ts) & (data.index <= end_ts)]

    daily = pd.DataFrame(index=data.index)
    daily["Open"] = data["Close"]
    daily["High"] = data["Close"]
    daily["Low"] = data["Close"]
    daily["Close"] = data["Close"]
    daily["Volume"] = 0.0

    if len(daily) < 300:
        raise ValueError(
            f"Fewer than 300 valid FRED {asset_label} rows returned: {len(daily)}"
        )

    return daily[REQUIRED_OHLCV_COLUMNS]


def load_fred_series(
    series_id: str,
    start: str,
    end: str,
    base_url: str = FRED_GRAPH_BASE_URL,
) -> pd.Series:
    """Load a raw FRED macro series (e.g. WALCL, FEDFUNDS, M2SL) as a date-indexed
    Series. No look-ahead handling here — the caller is responsible for lagging by
    the series' real publication delay."""
    query = urlencode({"id": series_id})
    csv_text = _fetch_text(f"{base_url}?{query}", source_name="FRED")
    data = pd.read_csv(StringIO(csv_text))
    _require_columns(data, ["observation_date", series_id])

    data["Date"] = pd.to_datetime(data["observation_date"])
    values = pd.to_numeric(data[series_id].replace(".", pd.NA), errors="coerce")
    series = pd.Series(values.to_numpy(), index=data["Date"]).dropna().sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    series = series[(series.index >= start_ts) & (series.index <= end_ts)]
    if series.empty:
        raise ValueError(f"FRED series {series_id} returned no data for {start}..{end}")
    return series


# Currency-ETF -> OECD 3-month interbank-rate FRED series. The home country's
# short rate is the carry proxy; carry_i = rate_i - rate_USD.
FX_CARRY_RATE_SERIES = {
    "FXE": "IR3TIB01DEM156N",  # EUR (Germany)
    "FXY": "IR3TIB01JPM156N",  # JPY
    "FXB": "IR3TIB01GBM156N",  # GBP
    "FXF": "IR3TIB01CHM156N",  # CHF (Switzerland)
    "FXA": "IR3TIB01AUM156N",  # AUD
    "FXC": "IR3TIB01CAM156N",  # CAD
}
FX_CARRY_USD_SERIES = "IR3TIB01USM156N"
# OECD rates are stamped at month-start and represent that whole month, published
# with a delay. A 60-day availability lag keeps the signal safely no-look-ahead
# (a month-start value is invisible until ~2 months later). Carry is slow-moving,
# so over-lagging costs little and is the honest default.
FX_CARRY_PUBLICATION_LAG_DAYS = 60


def _carry_panel_from_rates(
    rates_by_asset: dict[str, pd.Series],
    usd_rate: pd.Series,
    target_index: pd.DatetimeIndex,
    lag_days: int,
) -> pd.DataFrame:
    """Turn monthly interest-rate series into a daily carry (rate-differential)
    panel aligned to `target_index`.

    Each series is shifted forward by `lag_days` (its availability date), then
    forward-filled onto `target_index`, so any target date sees only rates whose
    availability is on or before it. carry[asset] = rate_asset - rate_usd. Pure:
    no network."""

    def _avail_daily(series: pd.Series) -> pd.Series:
        s = series.dropna().sort_index()
        s.index = s.index + pd.Timedelta(days=lag_days)
        return s.reindex(target_index, method="ffill")

    usd_daily = _avail_daily(usd_rate)
    columns = {
        asset: _avail_daily(series) - usd_daily
        for asset, series in rates_by_asset.items()
    }
    return pd.DataFrame(columns, index=target_index)


def load_fx_carry_rates(
    assets: list[str],
    start: str,
    end: str,
    target_index: pd.DatetimeIndex,
    lag_days: int = FX_CARRY_PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Load the daily carry (rate-differential vs USD) panel for currency-ETF
    `assets`, aligned to `target_index`. Rates are pulled from FRED far enough
    before `start` to cover the lag + smoothing warmup."""
    unknown = [a for a in assets if a.strip().upper() not in FX_CARRY_RATE_SERIES]
    if unknown:
        raise ValueError(
            "fx_carry requires currency ETFs with a known rate series; "
            f"unmapped: {unknown}. Supported: {sorted(FX_CARRY_RATE_SERIES)}"
        )
    # Pull rates from well before `start` so the lagged/ffilled panel is populated
    # at the backtest's first bar.
    rate_start = (pd.Timestamp(start) - pd.Timedelta(days=lag_days + 400)).date().isoformat()
    usd_rate = load_fred_series(FX_CARRY_USD_SERIES, rate_start, end)
    rates_by_asset = {
        asset: load_fred_series(FX_CARRY_RATE_SERIES[asset.strip().upper()], rate_start, end)
        for asset in assets
    }
    return _carry_panel_from_rates(rates_by_asset, usd_rate, target_index, lag_days)


def load_ohlcv_coingecko(
    asset: str,
    start: str,
    end: str,
    vs_currency: str = "usd",
    base_url: str = COINGECKO_PUBLIC_BASE_URL,
) -> pd.DataFrame:
    coin_id = _coingecko_coin_id_for_asset(asset)
    payload = _fetch_coingecko_market_chart_range(
        coin_id=coin_id,
        start=start,
        end=end,
        vs_currency=vs_currency,
        base_url=base_url,
    )
    data = _coingecko_market_chart_to_daily_ohlcv(payload)

    if len(data) < 300:
        raise ValueError(
            f"Fewer than 300 valid CoinGecko OHLC rows returned for {asset}: {len(data)}"
        )

    return data[REQUIRED_OHLCV_COLUMNS]


def load_ohlcv_coinmetrics_btc(
    start: str,
    end: str,
    base_url: str = COINMETRICS_COMMUNITY_BASE_URL,
) -> pd.DataFrame:
    prices = fetch_btc_coinmetrics_price_usd(start=start, base_url=base_url)
    if not prices:
        raise ValueError("Coin Metrics returned no BTC PriceUSD data")

    data = _coinmetrics_price_usd_to_daily_bars(prices, start=start, end=end)
    if len(data) < 300:
        raise ValueError(
            f"Fewer than 300 valid Coin Metrics BTC rows returned: {len(data)}"
        )

    return data[REQUIRED_OHLCV_COLUMNS]


def fetch_btc_coinmetrics_price_usd(
    start: str = "2010-07-17",
    base_url: str = COINMETRICS_COMMUNITY_BASE_URL,
) -> list[dict[str, int | float]]:
    url: str | None = _coinmetrics_asset_metrics_url(
        base_url=base_url,
        metrics=["PriceUSD"],
        start=start,
    )
    out: list[dict[str, int | float]] = []
    safety = 6

    while url and safety > 0:
        safety -= 1
        payload = _fetch_json(url, source_name="Coin Metrics")
        for row in payload.get("data", []):
            raw_time = row.get("time")
            raw_price = row.get("PriceUSD")
            if raw_time is None or raw_price is None:
                continue
            price = _parse_float(raw_price)
            if price is None:
                continue
            timestamp = int(pd.Timestamp(raw_time, tz="UTC").timestamp())
            out.append({"t": timestamp, "v": price})
        url = payload.get("next_page_url")

    return sorted(out, key=lambda point: int(point["t"]))


def is_bitcoin_asset(asset: str) -> bool:
    return asset.strip().lower() in {"btc", "btc-usd", "xbt", "xbt-usd", "bitcoin"}


def is_nasdaq_proxy_asset(asset: str) -> bool:
    return asset.strip().lower() in {
        "qqq",
        "nasdaq",
        "nasdaq100",
        "nasdaq-100",
        "ndx",
        "nasdaq100",
    }


def load_ohlcv_for_asset(asset: str, start: str, end: str) -> pd.DataFrame:
    """Auto-route a single asset to the right source by symbol, mirroring the
    single-asset run_backtest dispatch but reusable for portfolio loading.

    BTC -> Coin Metrics, Nasdaq proxies -> FRED, everything else -> Tiingo (when
    TIINGO_API_KEY is set) falling back to yfinance then Stooq.
    """
    if is_bitcoin_asset(asset):
        return load_ohlcv_coinmetrics_btc(start, end)
    if is_nasdaq_proxy_asset(asset):
        return load_ohlcv_fred_nasdaq(asset, start, end)
    source_notes: list[str] = []
    if os.getenv("TIINGO_API_KEY"):
        try:
            return load_ohlcv_tiingo(asset, start, end)
        except ValueError as tiingo_error:
            if _is_terminal_tiingo_error(tiingo_error):
                raise ValueError(
                    f"{asset}: Tiingo failed and fallback was not attempted because "
                    f"the error is quota/auth related: {tiingo_error}"
                ) from tiingo_error
            source_notes.append(f"Tiingo failed: {tiingo_error}")
    else:
        source_notes.append(
            "Tiingo skipped because TIINGO_API_KEY is not set. Add it to .env "
            "for US ETF data; Yahoo/yfinance is only a fallback and often fails "
            "with TLS/curl errors on some Linux systems."
        )
    try:
        return load_ohlcv_yfinance(asset, start, end)
    except ValueError as yfinance_error:
        try:
            return load_ohlcv_stooq(asset, start, end)
        except ValueError as stooq_error:
            source_notes.append(f"yfinance failed: {yfinance_error}")
            source_notes.append(f"Stooq fallback failed: {stooq_error}")
            raise ValueError(
                f"{asset}: " + " | ".join(source_notes)
            ) from stooq_error


def _is_terminal_tiingo_error(error: ValueError) -> bool:
    message = str(error).lower()
    terminal_markers = (
        "http 401",
        "http 403",
        "http 429",
        "hourly request allocation",
        "api key",
        "api token",
        "forbidden",
        "unauthorized",
    )
    return any(marker in message for marker in terminal_markers)


def load_portfolio_panel(
    assets: list[str], start: str, end: str, min_rows: int = 300
) -> pd.DataFrame:
    """Load each asset's daily Close and align them on a common date index.

    Returns a (dates x assets) DataFrame of close prices, columns ordered as
    given. Rows where any asset is missing are dropped so momentum/rank logic
    never sees NaN gaps. Raises if fewer than `min_rows` common dates remain.
    """
    if len(assets) < 2:
        raise ValueError("A portfolio panel needs at least 2 assets")

    closes: dict[str, pd.Series] = {}
    failures: list[str] = []
    for asset in assets:
        try:
            frame = load_ohlcv_for_asset(asset, start, end)
        except Exception as exc:  # noqa: BLE001 - surface a combined message below
            failures.append(f"{asset}: {exc}")
            continue
        series = frame["Close"].astype(float)
        series.index = pd.to_datetime(series.index)
        closes[asset] = series

    if failures:
        raise ValueError("Could not load all portfolio assets. " + " | ".join(failures))

    panel = pd.DataFrame(closes)
    panel = panel.dropna(how="any").sort_index()
    panel = panel[[asset for asset in assets if asset in panel.columns]]

    if len(panel) < min_rows:
        raise ValueError(
            f"Only {len(panel)} common dates across {assets}; "
            f"need at least {min_rows}. Assets may have non-overlapping histories."
        )
    return panel


def _stooq_symbol(ticker: str) -> str:
    normalized = ticker.strip().lower()
    aliases = {
        "nasdaq": "qqq.us",
        "nasdaq100": "qqq.us",
        "nasdaq-100": "qqq.us",
        "ndx": "qqq.us",
    }
    if normalized in aliases:
        return aliases[normalized]
    if "." in normalized or normalized.startswith("^"):
        return normalized
    return f"{normalized}.us"


def _fred_nasdaq_series_id(asset: str) -> str:
    if is_nasdaq_proxy_asset(asset):
        return "NASDAQ100"
    raise ValueError(f"FRED Nasdaq loader only supports Nasdaq proxy assets, got: {asset}")


def _normalize_yfinance_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    normalized = data.copy()
    for level in range(normalized.columns.nlevels):
        values = {str(value) for value in normalized.columns.get_level_values(level)}
        if ticker in values:
            normalized = normalized.xs(ticker, axis=1, level=level)
            break

    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [
            next((part for part in col if part in REQUIRED_OHLCV_COLUMNS + ["Adj Close"]), col[-1])
            for col in normalized.columns
        ]

    return normalized


def _coingecko_coin_id_for_asset(asset: str) -> str:
    if is_bitcoin_asset(asset):
        return "bitcoin"
    raise ValueError(
        f"CoinGecko loader only supports BTC assets in the MVP, got: {asset}"
    )


def _fetch_coingecko_market_chart_range(
    coin_id: str,
    start: str,
    end: str,
    vs_currency: str,
    base_url: str,
) -> dict:
    query = urlencode(
        {
            "vs_currency": vs_currency,
            "from": _start_of_day_unix(start),
            "to": _end_of_day_unix(end),
        }
    )
    url = f"{base_url.rstrip('/')}/coins/{coin_id}/market_chart/range?{query}"
    request = Request(url, headers={"accept": "application/json"})

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"CoinGecko request failed with HTTP {exc.code}: {body[:300]}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"CoinGecko request failed: {exc}") from exc


def _coinmetrics_asset_metrics_url(
    base_url: str,
    metrics: list[str],
    start: str,
) -> str:
    query = urlencode(
        {
            "assets": "btc",
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start,
            "page_size": 10000,
        }
    )
    return f"{base_url.rstrip('/')}/timeseries/asset-metrics?{query}"


def _fetch_json(url: str, source_name: str) -> dict:
    request = Request(url, headers={"accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"{source_name} request failed with HTTP {exc.code}: {body[:300]}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"{source_name} request failed: {exc}") from exc


def _fetch_text(url: str, source_name: str) -> str:
    request = Request(url, headers={"accept": "text/csv"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"{source_name} request failed with HTTP {exc.code}: {body[:300]}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(f"{source_name} request failed: {exc}") from exc


def _coinmetrics_price_usd_to_daily_bars(
    prices: list[dict[str, int | float]],
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(prices)
    frame["Date"] = pd.to_datetime(frame["t"], unit="s", utc=True)
    frame["Close"] = pd.to_numeric(frame["v"], errors="coerce")
    frame = frame.dropna(subset=["Close"]).set_index("Date").sort_index()

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    frame = frame[(frame.index >= start_ts) & (frame.index <= end_ts)]

    daily = pd.DataFrame(index=frame.index)
    daily["Open"] = frame["Close"]
    daily["High"] = frame["Close"]
    daily["Low"] = frame["Close"]
    daily["Close"] = frame["Close"]
    daily["Volume"] = 0.0
    daily.index = daily.index.tz_localize(None)
    return daily


def _coingecko_market_chart_to_daily_ohlcv(payload: dict) -> pd.DataFrame:
    prices = payload.get("prices", [])
    if not prices:
        raise ValueError("CoinGecko returned no price data")

    price_frame = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
    price_frame["Date"] = pd.to_datetime(price_frame["timestamp_ms"], unit="ms", utc=True)
    price_frame = price_frame.set_index("Date").sort_index()

    daily = price_frame["price"].resample("1D").agg(
        Open="first",
        High="max",
        Low="min",
        Close="last",
    )

    volumes = payload.get("total_volumes", [])
    if volumes:
        volume_frame = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])
        volume_frame["Date"] = pd.to_datetime(
            volume_frame["timestamp_ms"], unit="ms", utc=True
        )
        volume_frame = volume_frame.set_index("Date").sort_index()
        daily["Volume"] = volume_frame["volume"].resample("1D").last()
    else:
        daily["Volume"] = 0.0

    daily = daily.dropna(subset=["Open", "High", "Low", "Close"])
    daily["Volume"] = daily["Volume"].fillna(0.0)
    daily.index = daily.index.tz_localize(None)
    return daily


def _parse_float(value: str | int | float) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _start_of_day_unix(value: str) -> int:
    parsed = datetime.combine(datetime.fromisoformat(value).date(), time.min, tzinfo=UTC)
    return int(parsed.timestamp())


def _end_of_day_unix(value: str) -> int:
    parsed = datetime.combine(datetime.fromisoformat(value).date(), time.max, tzinfo=UTC)
    return int(parsed.timestamp())


def _require_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")


def _use_adjusted_prices_if_available(data: pd.DataFrame) -> pd.DataFrame:
    adjusted = data.copy()
    if "Adj Close" not in adjusted.columns:
        return adjusted

    close = adjusted["Close"].replace(0, pd.NA)
    adjustment_ratio = adjusted["Adj Close"] / close
    for column in ["Open", "High", "Low"]:
        adjusted[column] = adjusted[column] * adjustment_ratio
    adjusted["Close"] = adjusted["Adj Close"]
    return adjusted
