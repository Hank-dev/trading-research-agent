"""Canonical EMA + Donchian pullback + ATR risk strategy.

This module is intentionally plain pandas/numpy so every research script can
share one signal and execution path. Signals are decided on a completed bar and
filled at the next bar's open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendPullbackParams:
    side: str = "long"
    ema_period: int = 150
    donchian_window: int = 21
    delay_bars: int = 2
    atr_window: int = 20
    sl_atr: float = 1.0
    trail_atr: float = 3.0
    cost_bps_per_side: float = 0.0


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Average True Range using a simple rolling mean."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=window).mean()


def prepare_indicators(data: pd.DataFrame, params: TrendPullbackParams) -> pd.DataFrame:
    """Return OHLC data with canonical indicators and raw entry signal."""
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"OHLC data missing required columns: {sorted(missing)}")

    out = data[["Open", "High", "Low", "Close"]].astype(float).dropna().copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["ema"] = close.ewm(
        span=params.ema_period,
        adjust=False,
        min_periods=params.ema_period,
    ).mean()
    out["atr"] = atr(high, low, close, params.atr_window)

    if params.side == "long":
        channel = low.rolling(
            params.donchian_window,
            min_periods=params.donchian_window,
        ).min()
        touched = low.shift(params.delay_bars) <= channel.shift(params.delay_bars)
        recovered = close > channel
        trend = close > out["ema"]
    elif params.side == "short":
        channel = high.rolling(
            params.donchian_window,
            min_periods=params.donchian_window,
        ).max()
        touched = high.shift(params.delay_bars) >= channel.shift(params.delay_bars)
        recovered = close < channel
        trend = close < out["ema"]
    else:
        raise ValueError("side must be 'long' or 'short'")

    out["channel"] = channel
    out["raw_entry"] = (trend & recovered & touched).fillna(False)
    return out


def simulate_trades(data: pd.DataFrame, params: TrendPullbackParams) -> list[dict]:
    """Simulate trades using completed-bar signals and next-open fills.

    The trailing stop is updated only after checking the current bar against the
    previously known stop. That avoids using the same bar's close to create a
    stop that could also be hit earlier inside that bar.
    """
    prepared = prepare_indicators(data, params)
    o = prepared["Open"].to_numpy(float)
    h = prepared["High"].to_numpy(float)
    lo = prepared["Low"].to_numpy(float)
    c = prepared["Close"].to_numpy(float)
    atr_values = prepared["atr"].to_numpy(float)
    raw_entry = prepared["raw_entry"].to_numpy(bool)
    index = prepared.index

    trades: list[dict] = []
    in_position = False
    entry_price = 0.0
    entry_i = 0
    entry_atr = 0.0
    hard_stop = 0.0
    trail_stop = 0.0
    anchor_close = 0.0
    entry_signal_i = 0

    cost_pct = params.cost_bps_per_side / 100.0

    for i in range(len(prepared)):
        if in_position:
            exit_level = hard_stop
            if params.side == "long":
                exit_level = max(hard_stop, trail_stop)
                hit = lo[i] <= exit_level
                if hit:
                    exit_price = min(o[i], exit_level) if np.isfinite(o[i]) else exit_level
                    gross_pct = (exit_price - entry_price) / entry_price * 100.0
                else:
                    if c[i] > anchor_close:
                        anchor_close = c[i]
                        trail_stop = max(trail_stop, anchor_close - params.trail_atr * entry_atr)
            else:
                exit_level = min(hard_stop, trail_stop)
                hit = h[i] >= exit_level
                if hit:
                    exit_price = max(o[i], exit_level) if np.isfinite(o[i]) else exit_level
                    gross_pct = (entry_price - exit_price) / entry_price * 100.0
                else:
                    if c[i] < anchor_close:
                        anchor_close = c[i]
                        trail_stop = min(trail_stop, anchor_close + params.trail_atr * entry_atr)

            if hit:
                net_pct = gross_pct - 2.0 * cost_pct
                risk_pct = params.sl_atr * entry_atr / entry_price * 100.0
                trades.append(
                    {
                        "entry_date": index[entry_i],
                        "exit_date": index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return_pct": gross_pct,
                        "net_return_pct": net_pct,
                        "r_multiple": net_pct / risk_pct if risk_pct else np.nan,
                        "bars_held": i - entry_i,
                        "entry_signal_date": index[entry_signal_i],
                        "exit_reason": "stop_or_trail",
                    }
                )
                in_position = False
                continue

        if in_position:
            continue

        if not raw_entry[i] or i + 1 >= len(prepared):
            continue
        fill_i = i + 1
        fill_price = o[fill_i]
        signal_atr = atr_values[i]
        if not (np.isfinite(fill_price) and np.isfinite(signal_atr)):
            continue
        if fill_price <= 0 or signal_atr <= 0:
            continue

        in_position = True
        entry_i = fill_i
        entry_signal_i = i
        entry_price = float(fill_price)
        entry_atr = float(signal_atr)
        anchor_close = entry_price
        if params.side == "long":
            hard_stop = entry_price - params.sl_atr * entry_atr
            trail_stop = entry_price - params.trail_atr * entry_atr
        else:
            hard_stop = entry_price + params.sl_atr * entry_atr
            trail_stop = entry_price + params.trail_atr * entry_atr

    if in_position:
        exit_price = c[-1]
        if params.side == "long":
            gross_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            gross_pct = (entry_price - exit_price) / entry_price * 100.0
        net_pct = gross_pct - 2.0 * cost_pct
        risk_pct = params.sl_atr * entry_atr / entry_price * 100.0
        trades.append(
            {
                "entry_date": index[entry_i],
                "exit_date": index[-1],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_pct": gross_pct,
                "net_return_pct": net_pct,
                "r_multiple": net_pct / risk_pct if risk_pct else np.nan,
                "bars_held": len(prepared) - 1 - entry_i,
                "entry_signal_date": index[entry_signal_i],
                "exit_reason": "end_of_data",
            }
        )

    return trades


def summarize_trades(trades: list[dict]) -> dict:
    """Summarize a trade list."""
    if not trades:
        return {
            "trades": 0,
            "net_return_pct": 0.0,
            "gross_return_pct": 0.0,
            "total_r": 0.0,
            "avg_r": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_dd_pct": 0.0,
            "avg_hold_bars": np.nan,
        }

    returns = np.array([t["net_return_pct"] for t in trades], dtype=float)
    gross = np.array([t["gross_return_pct"] for t in trades], dtype=float)
    r_values = np.array([t["r_multiple"] for t in trades], dtype=float)
    equity = np.cumprod(1.0 + returns / 100.0)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100.0
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    pf = np.inf if losses == 0 and gains > 0 else (gains / losses if losses > 0 else np.nan)
    return {
        "trades": int(len(trades)),
        "net_return_pct": float((equity[-1] - 1.0) * 100.0),
        "gross_return_pct": float((np.cumprod(1.0 + gross / 100.0)[-1] - 1.0) * 100.0),
        "total_r": float(np.nansum(r_values)),
        "avg_r": float(np.nanmean(r_values)),
        "win_rate": float((returns > 0).mean() * 100.0),
        "profit_factor": float(pf) if np.isfinite(pf) else np.inf,
        "max_dd_pct": float(dd.min()),
        "avg_hold_bars": float(np.mean([t["bars_held"] for t in trades])),
    }


def split_periods(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice data by date while preserving index type."""
    return data.loc[pd.to_datetime(start) : pd.to_datetime(end)].copy()
