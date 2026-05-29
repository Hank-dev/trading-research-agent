from dataclasses import dataclass
import os
from typing import Any

import pandas as pd

from trading_research_agent.tools.data_loader import (
    is_bitcoin_asset,
    is_nasdaq_proxy_asset,
    load_ohlcv_for_asset,
    tiingo_cache_status,
)


@dataclass(frozen=True)
class AssetHealth:
    asset: str
    source: str
    cache: str
    status: str
    rows: int
    first_date: str | None
    last_date: str | None
    detail: str


def check_data_health(
    assets: list[str],
    start: str,
    end: str,
    *,
    min_rows: int = 300,
) -> dict[str, Any]:
    """Preflight data availability for a portfolio universe.

    This intentionally uses the real loader so successful checks populate the
    local cache. If Tiingo reports quota/auth failure, subsequent uncached Tiingo
    assets are skipped instead of repeating doomed requests.
    """
    if len(assets) < 1:
        raise ValueError("data health requires at least one asset")
    if not start or not end:
        raise ValueError("data health requires --start and --end")

    frames: dict[str, pd.DataFrame] = {}
    checks: list[AssetHealth] = []
    terminal_tiingo_error: str | None = None

    for asset in assets:
        source = _source_for_asset(asset)
        cache = _cache_label(asset, source, start, end)
        if terminal_tiingo_error and source == "tiingo" and cache != "covered":
            checks.append(
                AssetHealth(
                    asset=asset,
                    source=source,
                    cache=cache,
                    status="error",
                    rows=0,
                    first_date=None,
                    last_date=None,
                    detail=f"Skipped after Tiingo quota/auth failure: {terminal_tiingo_error}",
                )
            )
            continue

        try:
            frame = load_ohlcv_for_asset(asset, start, end)
        except Exception as exc:  # noqa: BLE001 - health check must report all assets.
            detail = str(exc)
            if source == "tiingo" and _is_tiingo_quota_or_auth_error(detail):
                terminal_tiingo_error = detail
            checks.append(
                AssetHealth(
                    asset=asset,
                    source=source,
                    cache=cache,
                    status="error",
                    rows=0,
                    first_date=None,
                    last_date=None,
                    detail=detail,
                )
            )
            continue

        frame = frame.sort_index()
        frames[asset] = frame
        checks.append(
            AssetHealth(
                asset=asset,
                source=source,
                cache=_cache_label(asset, source, start, end),
                status="ok",
                rows=len(frame),
                first_date=str(frame.index.min().date()) if len(frame) else None,
                last_date=str(frame.index.max().date()) if len(frame) else None,
                detail="OK",
            )
        )

    panel_rows = 0
    panel_start = None
    panel_end = None
    if frames and len(frames) == len(assets):
        panel = pd.DataFrame(
            {asset: frames[asset]["Close"].astype(float) for asset in assets}
        ).dropna(how="any")
        panel_rows = len(panel)
        if panel_rows:
            panel_start = str(panel.index.min().date())
            panel_end = str(panel.index.max().date())

    errors = [check for check in checks if check.status != "ok"]
    runnable = not errors and panel_rows >= min_rows
    reason = "OK"
    if errors:
        reason = "; ".join(f"{check.asset}: {check.detail}" for check in errors)
    elif panel_rows < min_rows:
        reason = (
            f"Only {panel_rows} common aligned rows; need at least {min_rows}. "
            "Assets may have non-overlapping histories."
        )

    return {
        "assets": assets,
        "start": start,
        "end": end,
        "checks": checks,
        "common_rows": panel_rows,
        "common_start": panel_start,
        "common_end": panel_end,
        "min_rows": min_rows,
        "runnable": runnable,
        "reason": reason,
    }


def _source_for_asset(asset: str) -> str:
    if is_bitcoin_asset(asset):
        return "coinmetrics"
    if is_nasdaq_proxy_asset(asset):
        return "fred"
    if os.getenv("TIINGO_API_KEY"):
        return "tiingo"
    return "yfinance/stooq"


def _cache_label(asset: str, source: str, start: str, end: str) -> str:
    if source != "tiingo":
        return "n/a"
    info = tiingo_cache_status(asset, start, end)
    if not info["enabled"]:
        return "disabled"
    return "covered" if info["covered"] else "missing"


def _is_tiingo_quota_or_auth_error(detail: str) -> bool:
    lowered = detail.lower()
    markers = (
        "http 401",
        "http 403",
        "http 429",
        "quota/auth",
        "hourly request allocation",
        "unauthorized",
        "forbidden",
    )
    return any(marker in lowered for marker in markers)
