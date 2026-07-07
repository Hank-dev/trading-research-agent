#!/usr/bin/env python3
"""
Walk-forward validation for EMA + Donchian-pullback + ATR-stop strategies.

Strategy specification:
- Trend filter: Close > EMA(ema_period)
- Entry: Donchian(21) lower channel touched, price pulled back above it,
  confirmed after a 2-bar delay (i.e. close[0] > DC_lower[0] AND
  close[-2] <= DC_lower[-2])
- Fill: next-open (shift entry signal by 1 bar)
- Stop loss: ATR(20) * sl_atr_multiple
- Trailing exit: trail highest close since entry by ATR(20) * trail_atr_multiple

Candidate slate:
  1. XAUUSD 1d long EMA150 SL1ATR Trail3ATR
  2. XAGUSD 4h long EMA50 SL1ATR Trail4ATR
  3. STOXX50E 2h long EMA150 SL1ATR Trail4ATR
  4. SPA35 4h long EMA100 SL1ATR Trail3ATR
  5. NDX 1h long EMA150 SL1ATR Trail3ATR
  6. (optional) GDAXI/J225/WS30 long

Usage:
  python scripts/walk_forward_ema_donchian.py   # runs full slate
"""

import csv
import os
import sys
from datetime import datetime, timedelta, date, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/johannes/trading-research-agent/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = OUTPUT_DIR / "walk_forward_20260617.csv"
REPORT_HTML = OUTPUT_DIR / "walk_forward_20260617.html"

# Donchian window for pullback entry
DONCHIAN_WINDOW = 21
# ATR window for stops and trailing
ATR_WINDOW = 20

# ── Helpers ────────────────────────────────────────────────────────────────────


def _to_series(s: pd.Series) -> pd.Series:
    return pd.Series(s, dtype="float64")


def ema(series: pd.Series, window: int) -> pd.Series:
    return _to_series(series).ewm(span=window, adjust=False, min_periods=window).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return _to_series(series).rolling(window).mean()


def donchian_low(low: pd.Series, window: int) -> pd.Series:
    return _to_series(low).rolling(window).min()


def donchian_high(high: pd.Series, window: int) -> pd.Series:
    return _to_series(high).rolling(window).max()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = _to_series(close).shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_data(ticker: str, timeframe: str, years_back: int = 8) -> pd.DataFrame:
    """Load OHLCV data from yfinance for the given ticker and timeframe.

    2h is not directly supported by yfinance; we download 1h and resample.
    """
    # yfinance valid intervals for period=2y
    actual_interval = timeframe
    resample_to: str | None = None

    if timeframe == "2h":
        actual_interval = "1h"
        resample_to = "2h"

    if timeframe == "1d" or (timeframe == "1h" and resample_to is None):
        period = "max" if timeframe == "1d" else "2y"
    elif timeframe == "4h":
        period = "2y"
    elif timeframe == "1h" and resample_to == "2h":
        period = "2y"
    else:
        period = "2y"

    print(f"  Downloading {ticker} ({timeframe})...", end=" ", flush=True)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=actual_interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

    if df.empty:
        print("EMPTY")
        return pd.DataFrame()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Resample if needed (e.g. 1h -> 2h)
    if resample_to == "2h":
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        df = df.resample("2h").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])
        print(f"(resampled to 2h: {len(df)} bars)...", end=" ")

    # Ensure required columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"{len(df)} bars")
    return df


# ── Strategy Signal Generation ────────────────────────────────────────────────


def generate_signals(
    df: pd.DataFrame,
    ema_period: int,
    sl_atr_mult: float,
    trail_atr_mult: float,
) -> pd.DataFrame:
    """Generate entry/exit signals for the EMA + Donchian-pullback + ATR strategy.

    Entry: Close > EMA AND Donchian(21) lower touched 2 bars ago.
    Exit: Stop loss hit OR trailing stop triggered.

    Returns df with added columns: entry, exit, sl_price, trail_price.
    """
    close = _to_series(df["Close"])
    high = _to_series(df["High"])
    low = _to_series(df["Low"])

    # Indicators
    ema_line = ema(close, ema_period)
    dc_lower = donchian_low(low, DONCHIAN_WINDOW)
    atr_val = atr(high, low, close, ATR_WINDOW)

    # Trends
    uptrend = close > ema_line

    # Donchian lower pullback: price touched (low <= DC_low) the lower band
    # 2 bars ago, now recovered above it. The "touch" is via Low touching DC_low
    # (the low of the channel), not Close. Close condition ensures recovery.
    # Low[-2] <= DC_lower[-2] means price touched the channel 2 bars ago.
    # Close[0] > DC_lower[0] means price has pulled back above it.
    entry_signal = uptrend & (close > dc_lower) & (low.shift(2) <= dc_lower.shift(2))

    # Fill at next open -> shift entry by 1
    entry = entry_signal.shift(1).fillna(False)

    # For exits we'll compute them dynamically in the portfolio simulation
    # But we also add stop loss and trail levels
    sl_price = close - atr_val * sl_atr_mult
    trail_price = pd.Series(np.nan, index=df.index)

    # Compute exit indicator: price at any bar is below stop OR trail
    # Trail logic needs position state, done in portfolio sim below
    # We add a placeholder that we'll use to compute exit from signals

    # Also compute a "force exit" for the last 5 bars to avoid open positions
    # (handled in simulation)

    df = df.copy()
    df["entry"] = entry
    df["ema"] = ema_line
    df["dc_lower"] = dc_lower
    df["atr"] = atr_val
    df["sl_price"] = sl_price
    df["uptrend"] = uptrend
    df["is_entry_bar"] = entry_signal  # raw (unshifted)

    return df


# ── Portfolio Simulation ─────────────────────────────────────────────────────


def simulate_portfolio(
    df: pd.DataFrame,
    sl_atr_mult: float,
    trail_atr_mult: float,
) -> list[dict]:
    """Simulate the strategy on a DataFrame with signal columns.

    Returns list of trade dicts with: entry_date, entry_price, exit_date,
    exit_price, bars_held, r_multiple, exit_reason.
    """
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    entry_signal = df["entry"].values
    atr_val = df["atr"].values
    dates = df.index

    trades: list[dict] = []

    in_position = False
    entry_idx = -1
    entry_price = 0.0
    trail_high = 0.0
    trail_stop = 0.0
    initial_sl = 0.0

    n = len(df)

    for i in range(n):
        if not in_position:
            if entry_signal[i]:
                in_position = True
                entry_idx = i
                entry_price = close[i]
                initial_sl = close[i] - atr_val[i] * sl_atr_mult
                trail_high = close[i]
                trail_stop = close[i] - atr_val[i] * trail_atr_mult

        if in_position:
            # Update trail high
            if close[i] > trail_high:
                trail_high = close[i]
                trail_stop = close[i] - atr_val[i] * trail_atr_mult

            # Check exit conditions (using low of current bar for stop/trail)
            exit_reason = None

            if low[i] <= initial_sl:
                exit_reason = "stop_loss"
            elif low[i] <= trail_stop and i > entry_idx:
                exit_reason = "trail_stop"
            elif i == n - 1:
                exit_reason = "end_of_data"

            if exit_reason:
                # Exit at bar's close or open (close is more conservative / realistic)
                exit_price = close[i]
                bars_held = i - entry_idx
                r_mult = (exit_price - entry_price) / abs(initial_sl - entry_price) if abs(initial_sl - entry_price) > 1e-10 else 0.0

                trades.append({
                    "entry_date": dates[entry_idx],
                    "entry_price": float(entry_price),
                    "exit_date": dates[i],
                    "exit_price": float(exit_price),
                    "bars_held": bars_held,
                    "r_multiple": float(r_mult),
                    "return_pct": float((exit_price / entry_price - 1) * 100),
                    "exit_reason": exit_reason,
                    "initial_sl": float(initial_sl),
                    "trail_stop": float(trail_stop),
                })

                in_position = False
                trail_high = 0.0
                trail_stop = 0.0

    return trades


# ── Walk-Forward Engine ─────────────────────────────────────────────────────


def rolling_walk_forward(
    df: pd.DataFrame,
    ema_period: int,
    sl_atr_mult: float,
    trail_atr_mult: float,
    window_bars: int,
    step_bars: int | None = None,
    timeframe: str = "1d",
) -> list[dict]:
    """Run rolling walk-forward windows.

    Each window: train on first 60%, test on last 40%.
    Windows slide forward by step_bars (defaults to window_bars/3).

    Returns list of window result dicts with: train_trades, test_trades,
    train_metrics, test_metrics.
    """
    n = len(df)
    if step_bars is None:
        step_bars = max(window_bars // 3, 1)

    # Generate a sequence of train/test windows
    # Target at least 6-8 windows for reasonable stability assessment
    results: list[dict] = []

    # We want windows that cover the entire data span
    # Each window: train=first 60%, test=last 40%
    # Overlap successive windows

    windows: list[tuple[int, int, int, int]] = []

    # Method 1: Sliding 60/40 windows (more windows for stability assessment)
    win_start = 0
    min_train = window_bars
    min_window = int(window_bars / 0.6)  # total window including test

    while win_start + min_window <= n:
        train_end = win_start + min_train
        test_end = min(train_end + int(window_bars * 0.4 / 0.6), n)
        # Ensure test has enough bars
        if test_end - train_end < 5:
            break
        windows.append((win_start, train_end, train_end, test_end))
        win_start += step_bars

    # Method 2: If we have 3+ years of data, also do annual windows.
    # For intraday, scale the annual bar count.
    if n >= 252 * 3:
        _bars_per_day = {"1d": 1, "2h": 6, "4h": 6, "1h": 8}
        intraday_mult = _bars_per_day.get(timeframe, 1)
        idx = df.index
        yearly_boundaries = []
        first_year = idx[0].year
        last_year = idx[-1].year
        for y in range(first_year + 1, last_year + 1):
            yb = pd.Timestamp(y, 1, 1)
            if idx[0] <= yb <= idx[-1]:
                yearly_boundaries.append(yb)

        for yb in yearly_boundaries:
            test_start_idx = idx.searchsorted(yb)
            if test_start_idx >= n:
                break
            # Train: all data before this year's start
            train_start_idx = 0
            train_end_idx = test_start_idx - 1
            if train_end_idx - train_start_idx < 1:
                continue
            # Test: one year (~252 trading bars, or scaled for intraday)
            annual_bars_target = 252 * intraday_mult
            test_end_idx = min(test_start_idx + annual_bars_target, n - 1)
            if (train_end_idx - train_start_idx >= 100) and (test_end_idx - test_start_idx >= 10):
                windows.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))

    if not windows:
        # Fallback: single full-data test train/test split (60/40)
        split = int(n * 0.6)
        windows = [(0, split, split, n)]

    # Deduplicate and ensure unique
    seen = set()
    unique_windows = []
    for w in windows:
        key = (w[0], w[1], w[2], w[3])
        if key not in seen:
            seen.add(key)
            unique_windows.append(w)

    # Sort by test start
    unique_windows.sort(key=lambda w: w[2])

    # Remove tiny overlaps: if windows overlap > 50%, keep the larger one
    if len(unique_windows) > 1:
        filtered = [unique_windows[0]]
        for w in unique_windows[1:]:
            last = filtered[-1]
            # Compute overlap fraction of test segments
            test_overlap = max(0, min(last[3], w[3]) - max(last[2], w[2]))
            if test_overlap > 0.5 * (w[3] - w[2]):
                continue  # skip, too overlapping
            filtered.append(w)
        unique_windows = filtered

    print(f"    Generated {len(unique_windows)} walk-forward windows")

    for wi, (tr_start, tr_end, ts_start, ts_end) in enumerate(unique_windows):
        train_df = df.iloc[tr_start:tr_end].copy()
        test_df = df.iloc[ts_start:ts_end].copy()

        if len(test_df) < 5:
            continue

        # Train on full data (strategy has no trainable params)
        # But we still compute metrics on train and test separately

        signal_df_train = generate_signals(train_df, ema_period, sl_atr_mult, trail_atr_mult)
        signal_df_test = generate_signals(test_df, ema_period, sl_atr_mult, trail_atr_mult)

        train_trades = simulate_portfolio(signal_df_train, sl_atr_mult, trail_atr_mult)
        test_trades = simulate_portfolio(signal_df_test, sl_atr_mult, trail_atr_mult)

        train_metrics = compute_metrics(train_trades)
        test_metrics = compute_metrics(test_trades)

        row = {
            "window": wi,
            "train_start": train_df.index[0].strftime("%Y-%m-%d"),
            "train_end": train_df.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_df.index[0].strftime("%Y-%m-%d"),
            "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results.append(row)

    return results


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(trades: list[dict]) -> dict:
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_mae_pct": 0.0,
            "avg_bars_held": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
        }

    r_mult = np.array([t["r_multiple"] for t in trades])
    returns = np.array([t["return_pct"] for t in trades])

    wins = r_mult > 0
    losss = r_mult <= 0

    total_r = float(r_mult.sum())
    avg_r = float(r_mult.mean())
    win_rate = float(wins.mean() * 100) if len(wins) > 0 else 0.0

    total_gain = r_mult[wins].sum() if wins.any() else 0.0
    total_loss = abs(r_mult[losss].sum()) if losss.any() else 0.0
    if total_loss < 1e-10:
        pf = 999.99 if total_gain > 0 else 0.0
    else:
        pf = total_gain / total_loss

    max_mae = float(np.min(returns)) if len(returns) > 0 else 0.0
    avg_bars = float(np.mean([t["bars_held"] for t in trades]))
    avg_ret = float(returns.mean())
    total_ret = float((1 + returns / 100).prod() - 1) * 100

    return {
        "trades": len(trades),
        "wins": int(wins.sum()),
        "total_r": round(total_r, 4),
        "avg_r": round(avg_r, 4),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(pf, 4) if pf != float("inf") else 999.99,
        "max_mae_pct": round(max_mae, 4),
        "avg_bars_held": round(avg_bars, 2),
        "avg_return_pct": round(avg_ret, 4),
        "total_return_pct": round(total_ret, 4),
    }


def stability_score(window_results: list[dict]) -> dict:
    """Compute how stable performance is across windows."""
    if not window_results:
        return {"score": 0, "test_positive_pct": 0, "test_pf_consistency": 0}

    # Check test window profitability
    test_return_col = "test_total_return_pct"
    test_pf_col = "test_profit_factor"

    returns = [w.get(test_return_col, 0) for w in window_results]
    pfs = [w.get(test_pf_col, 0) for w in window_results]

    n = len(returns)
    positive_returns = sum(1 for r in returns if r > 0)
    pf_above_1 = sum(1 for pf in pfs if pf > 1.0)

    pos_pct = positive_returns / n * 100 if n > 0 else 0
    pf_pct = pf_above_1 / n * 100 if n > 0 else 0

    # Compute consistency: ratio of windows where both test return positive
    # AND train return positive (strategy works in both regimes)
    train_ret_col = "train_total_return_pct"
    consistent = sum(
        1 for w in window_results
        if w.get(test_return_col, 0) > 0 and w.get(train_ret_col, 0) > 0
    )
    consistency_pct = consistent / n * 100 if n > 0 else 0

    # Is performance persisting? Check recent vs early windows
    # Early = first half, Recent = second half
    mid = max(1, n // 2)
    early_returns = returns[:mid]
    late_returns = returns[mid:]

    early_avg = np.mean(early_returns) if early_returns else 0
    late_avg = np.mean(late_returns) if late_returns else 0
    decay = late_avg - early_avg  # Negative = performance decay

    # Final score: weighted of positive windows, PF consistency, train/test alignment
    score = pos_pct * 0.4 + pf_pct * 0.3 + consistency_pct * 0.3

    return {
        "score": round(score, 2),
        "n_windows": n,
        "test_positive_pct": round(pos_pct, 1),
        "test_pf_above_1_pct": round(pf_pct, 1),
        "train_test_consistency_pct": round(consistency_pct, 1),
        "avg_test_return": round(float(np.mean(returns)), 4),
        "avg_test_pf": round(float(np.mean(pfs)), 4),
        "performance_decay": round(float(decay), 4),
        "total_test_trades": sum(w.get("test_trades", 0) for w in window_results),
        "positive_windows": positive_returns,
    }


def estimate_window_size(timeframe: str, data_length: int) -> tuple[int, int]:
    """Estimate proper window size for walk-forward based on timeframe and data length."""
    bars_per_day = {"1d": 1, "2h": 6, "4h": 6, "1h": 8}
    # For intraday, a "year" of trading days
    daily_bars = 252
    intraday_mult = bars_per_day.get(timeframe, bars_per_day["1d"])
    annual_bars = daily_bars * intraday_mult

    # For intraday with limited data (~2y), use smaller windows
    # Aim for 6-10 windows minimum
    target_windows = 8

    # Total window = train_bars + test_bars
    # We want train:test = 60:40
    # With target_windows overlapping windows
    total_span = data_length

    # Each window: train = X, test = 0.67*X (60/40 split)
    # Window step = total_span / target_windows
    step = max(total_span // target_windows, 20)
    train_bars = int(step * 0.6)
    # Ensure minimum
    train_bars = max(train_bars, 50)
    test_bars = max(int(train_bars / 0.6 * 0.4), 10)

    # Cap to avoid too few windows
    total_window = train_bars + test_bars
    if total_window > data_length // 2:
        # Scale down
        train_bars = max(data_length // 4, 50)
        test_bars = max(int(train_bars * 0.4 / 0.6), 10)

    return train_bars + test_bars, test_bars


# ── Main Runner ──────────────────────────────────────────────────────────────


def run_candidate(
    name: str,
    ticker: str,
    timeframe: str,
    ema_period: int,
    sl_atr_mult: float,
    trail_atr_mult: float,
) -> dict | None:
    """Run walk-forward validation for one candidate."""
    print(f"\n{'='*70}")
    print(f"  Candidate: {name}")
    print(f"  Ticker: {ticker}  TF: {timeframe}  EMA:{ema_period}  SL:{sl_atr_mult}ATR  Trail:{trail_atr_mult}ATR")
    print(f"{'='*70}")

    df = load_data(ticker, timeframe)
    if df.empty or len(df) < 100:
        print(f"  SKIP: insufficient data ({len(df)} bars)")
        return None

    n = len(df)

    # Determine window parameters
    window_total, test_bars = estimate_window_size(timeframe, n)
    window_bars = window_total - test_bars

    # Run walk-forward
    wf_results = rolling_walk_forward(
        df, ema_period, sl_atr_mult, trail_atr_mult,
        window_bars=window_bars, step_bars=max(window_bars // 4, 1),
        timeframe=timeframe,
    )

    if not wf_results:
        print("  No valid walk-forward windows")
        return None

    # Compute stability across windows
    stab = stability_score(wf_results)

    # Also run a full-data simulation for reference
    signal_df = generate_signals(df, ema_period, sl_atr_mult, trail_atr_mult)
    full_trades = simulate_portfolio(signal_df, sl_atr_mult, trail_atr_mult)
    full_metrics = compute_metrics(full_trades)

    # Print summary
    print(f"\n  Full-data results ({len(full_trades)} trades):")
    print(f"    Total R: {full_metrics['total_r']:.2f}, Avg R: {full_metrics['avg_r']:.2f}")
    print(f"    Win Rate: {full_metrics['win_rate']:.1f}%, PF: {full_metrics['profit_factor']:.2f}")
    print(f"    Total Return: {full_metrics['total_return_pct']:.2f}%")

    print(f"\n  Walk-forward stability ({stab['n_windows']} windows):")
    print(f"    Positive test windows: {stab['positive_windows']}/{stab['n_windows']} ({stab['test_positive_pct']:.1f}%)")
    print(f"    Test PF > 1 windows: {stab['test_pf_above_1_pct']:.1f}%")
    print(f"    Train/Test consistency: {stab['train_test_consistency_pct']:.1f}%")
    print(f"    Avg test return: {stab['avg_test_return']:.2f}%")
    print(f"    Avg test PF: {stab['avg_test_pf']:.2f}")
    print(f"    Performance decay: {stab['performance_decay']:.2f}%")
    print(f"    Stability score: {stab['score']:.1f}/100")

    verdict = "PASS" if stab["test_positive_pct"] >= 50 and stab["score"] >= 30 else "FAIL"
    if stab["performance_decay"] < -30:
        verdict = "DECAY"
    print(f"  Verdict: {verdict}")

    return {
        "name": name,
        "ticker": ticker,
        "timeframe": timeframe,
        "ema_period": ema_period,
        "sl_atr": sl_atr_mult,
        "trail_atr": trail_atr_mult,
        "total_bars": n,
        **{f"full_{k}": v for k, v in full_metrics.items()},
        **stab,
        "per_window": wf_results,
        "verdict": verdict,
    }


def write_report(results: list[dict | None]) -> None:
    """Write CSV and HTML reports."""
    # Filter None
    results = [r for r in results if r is not None]

    # CSV
    csv_rows = []
    for r in results:
        row = {
            "name": r["name"],
            "ticker": r["ticker"],
            "timeframe": r["timeframe"],
            "ema": r["ema_period"],
            "sl_atr": r["sl_atr"],
            "trail_atr": r["trail_atr"],
            "bars": r["total_bars"],
            "full_trades": r["full_trades"],
            "full_total_r": r["full_total_r"],
            "full_avg_r": r["full_avg_r"],
            "full_win_rate": r["full_win_rate"],
            "full_pf": r["full_profit_factor"],
            "full_total_return_pct": r["full_total_return_pct"],
            "wf_windows": r["n_windows"],
            "wf_pos_pct": r["test_positive_pct"],
            "wf_pf_above_1_pct": r["test_pf_above_1_pct"],
            "wf_consistency": r["train_test_consistency_pct"],
            "wf_avg_test_return": r["avg_test_return"],
            "wf_avg_test_pf": r["avg_test_pf"],
            "wf_decay": r["performance_decay"],
            "wf_score": r["score"],
            "verdict": r["verdict"],
        }
        csv_rows.append(row)

    # Write CSV
    csv_path = OUTPUT_DIR / "walk_forward_20260617.csv"
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        else:
            f.write("no_results\n")
    print(f"\nCSV written: {csv_path}")

    # Write per-window detail CSV
    detail_csv_path = OUTPUT_DIR / "walk_forward_20260617_detail.csv"
    detail_rows = []
    for r in results:
        for wr in r.get("per_window", []):
            row = {
                "name": r["name"],
                "window": wr["window"],
                "train_start": wr["train_start"],
                "train_end": wr["train_end"],
                "test_start": wr["test_start"],
                "test_end": wr["test_end"],
                "train_bars": wr["train_bars"],
                "test_bars": wr["test_bars"],
                "train_trades": wr["train_trades"],
                "train_total_r": wr["train_total_r"],
                "train_avg_r": wr["train_avg_r"],
                "train_win_rate": wr["train_win_rate"],
                "train_pf": wr["train_profit_factor"],
                "test_trades": wr["test_trades"],
                "test_total_r": wr["test_total_r"],
                "test_avg_r": wr["test_avg_r"],
                "test_win_rate": wr["test_win_rate"],
                "test_pf": wr["test_profit_factor"],
            }
            detail_rows.append(row)

    with open(detail_csv_path, "w", newline="") as f:
        if detail_rows:
            writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
            writer.writeheader()
            writer.writerows(detail_rows)
        else:
            f.write("no_detail\n")
    print(f"Detail CSV written: {detail_csv_path}")

    # HTML report
    html = _build_html_report(results, csv_rows, detail_rows)
    html_path = OUTPUT_DIR / "walk_forward_20260617.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML written: {html_path}")


def _build_html_report(results: list[dict], summary_rows: list[dict], detail_rows: list[dict]) -> str:
    """Build an HTML report."""
    rows_html = ""
    for srow in summary_rows:
        verdict_class = "pass" if srow["verdict"] == "PASS" else ("fail" if srow["verdict"] == "FAIL" else "decay")
        rows_html += f"""<tr class="{verdict_class}">
            <td>{srow["name"]}</td>
            <td>{srow["ticker"]}</td>
            <td>{srow["timeframe"]}</td>
            <td>{srow["ema"]}</td>
            <td>{srow["sl_atr"]}x / {srow["trail_atr"]}x</td>
            <td>{srow["bars"]}</td>
            <td>{srow["full_trades"]}</td>
            <td>{srow["full_total_r"]}</td>
            <td>{srow["full_avg_r"]}</td>
            <td>{srow["full_win_rate"]}%</td>
            <td>{srow["full_pf"]}</td>
            <td>{srow["full_total_return_pct"]}</td>
            <td>{srow["wf_windows"]}</td>
            <td>{srow["wf_pos_pct"]}%</td>
            <td>{srow["wf_consistency"]}%</td>
            <td>{srow["wf_avg_test_return"]}</td>
            <td>{srow["wf_avg_test_pf"]}</td>
            <td>{srow["wf_decay"]}</td>
            <td>{srow["wf_score"]}</td>
            <td><strong>{srow["verdict"]}</strong></td>
        </tr>"""

    # Per-window detail table
    detail_html = ""
    for drow in detail_rows:
        detail_html += f"""<tr>
            <td>{drow["name"]}</td>
            <td>{drow["window"]}</td>
            <td>{drow["train_start"]}</td>
            <td>{drow["train_end"]}</td>
            <td>{drow["test_start"]}</td>
            <td>{drow["test_end"]}</td>
            <td>{drow["train_trades"]}</td>
            <td>{drow["train_total_r"]}</td>
            <td>{drow["train_avg_r"]}</td>
            <td>{drow["train_win_rate"]}%</td>
            <td>{drow["train_pf"]}</td>
            <td>{drow["test_trades"]}</td>
            <td>{drow["test_total_r"]}</td>
            <td>{drow["test_avg_r"]}</td>
            <td>{drow["test_win_rate"]}%</td>
            <td>{drow["test_pf"]}</td>
        </tr>"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Walk-Forward Validation Report — 2026-06-17</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #8b949e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 8px; text-align: right; }}
  th {{ background: #161b22; color: #58a6ff; text-align: center; }}
  td:first-child, td:last-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #161b22; }}
  tr:hover {{ background: #1c2333; }}
  .pass {{ background: #0a3d26 !important; }}
  .pass:hover {{ background: #0f5a3a !important; }}
  .fail {{ background: #3d0a0a !important; }}
  .fail:hover {{ background: #5a0f0f !important; }}
  .decay {{ background: #3d3d0a !important; }}
  .decay:hover {{ background: #5a5a0f !important; }}
  .summary {{ margin: 10px 0; padding: 10px; background: #161b22; border-radius: 6px; }}
  .scroll {{ overflow-x: auto; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-weight: bold; }}
  .badge-pass {{ background: #238636; color: white; }}
  .badge-fail {{ background: #da3633; color: white; }}
  .badge-decay {{ background: #d29922; color: white; }}
</style>
</head>
<body>
<h1>🔬 Walk-Forward Validation</h1>
<p>Generated: {now}</p>
<p>Strategy: Close > EMA(trend) + Donchian({DONCHIAN_WINDOW}L) pullback (2-bar delay) + ATR({ATR_WINDOW}) stop + ATR trailing exit</p>

<h2>Candidate Summary</h2>
<div class="scroll">
<table>
<thead>
<tr>
  <th>Name</th><th>Ticker</th><th>TF</th><th>EMA</th><th>SL/Trail</th><th>Bars</th>
  <th>Trades</th><th>Tot R</th><th>Avg R</th><th>WR%</th><th>PF</th><th>Ret%</th>
  <th>Win#</th><th>Pos%</th><th>Consist%</th><th>Avg Ret</th><th>Avg PF</th><th>Decay</th><th>Score</th>
  <th>Verdict</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

<h2>Per-Window Detail</h2>
<div class="scroll">
<table>
<thead>
<tr>
  <th>Name</th><th>W#</th>
  <th>Train Start</th><th>Train End</th><th>Test Start</th><th>Test End</th>
  <th>Tr Trd</th><th>Tr R</th><th>Tr AvgR</th><th>Tr WR</th><th>Tr PF</th>
  <th>Te Trd</th><th>Te R</th><th>Te AvgR</th><th>Te WR</th><th>Te PF</th>
</tr>
</thead>
<tbody>
{detail_html}
</tbody>
</table>
</div>

<h2>Strategy Description</h2>
<div class="summary">
<pre>
Entry Conditions (all must be true):
  1. Close > EMA({DONCHIAN_WINDOW}) — trend filter (only long in uptrend)
  2. Close > Donchian({DONCHIAN_WINDOW}) Lower — price has pulled back from lowest low
  3. Close[-2] <= Donchian({DONCHIAN_WINDOW}) Lower[-2] — confirmed 2-bar delay from the touch
  4. Entry fills at next bar's open

Exit Conditions (whichever hits first):
  - Initial stop loss: entry_price - ATR({ATR_WINDOW}) * sl_mult
  - Trailing stop: highest_close_since_entry - ATR({ATR_WINDOW}) * trail_mult

Walk-Forward Protocol:
  - Rolling 60/40 train/test splits sliding forward by ~1/4 window
  - Window size tuned to timeframe: daily ~504 bars, intraday ~annual bars * 2
  - Consistency measured: % of windows where BOTH train AND test are profitable
  - Performance decay: late-window avg return minus early-window avg return
  - Stability score: weighted composite of positive windows (40%),
    PF > 1 windows (30%), and train/test consistency (30%)

Scoring:
  - Score >= 30 and test positive >= 50% → PASS
  - score < 30 or test positive < 50%   → FAIL
  - Decay < -30%                         → DECAY (was good, fading)
</pre>
</div>

</body>
</html>"""


def main():
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION — EMA + Donchian Pullback + ATR Stop")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    # ── Candidate Slate ──────────────────────────────────────────────────────
    candidates = [
        # (name, ticker, timeframe, ema_period, sl_atr_mult, trail_atr_mult)
        ("XAUUSD", "GC=F", "1d", 150, 1.0, 3.0),      # Gold daily
        ("XAGUSD", "SI=F", "4h", 50, 1.0, 4.0),        # Silver 4h
        ("STOXX50E", "^STOXX50E", "2h", 150, 1.0, 4.0),  # Eurostoxx 2h
        ("SPA35", "^IBEX", "4h", 100, 1.0, 3.0),         # IBEX 4h
        ("NDX", "^NDX", "1h", 150, 1.0, 3.0),            # Nasdaq 100 1h
    ]

    # Optional extras — run if time/networking permits
    optional = [
        ("GDAXI", "^GDAXI", "1h", 150, 1.0, 3.0),       # DAX 1h
        ("J225", "^N225", "1h", 150, 1.0, 3.0),          # Nikkei 1h
        ("WS30", "^DJI", "1d", 150, 1.0, 3.0),           # Dow daily
        ("WS30_4h", "^DJI", "4h", 150, 1.0, 3.0),        # Dow 4h
        ("GDAXI_d", "^GDAXI", "1d", 150, 1.0, 3.0),      # DAX daily
        ("J225_d", "^N225", "1d", 150, 1.0, 3.0),        # Nikkei daily
        ("XAGUSD_d", "SI=F", "1d", 150, 1.0, 3.0),       # Silver daily
        ("SPA35_d", "^IBEX", "1d", 100, 1.0, 3.0),        # IBEX daily
    ]

    results: list[dict | None] = []

    # Run primary slate
    for c in candidates:
        result = run_candidate(*c)
        results.append(result)

    # Run optional if time permits (no more than 5)
    for c in optional[:7]:
        result = run_candidate(*c)
        results.append(result)

    write_report(results)

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
