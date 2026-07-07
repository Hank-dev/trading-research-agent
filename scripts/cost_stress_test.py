#!/usr/bin/env python3
"""Cost stress test for ATR trend-pullback candidate slate.

Tests each candidate at 0, 2, 5, 10, 20 bps **per side** (entry + exit).
Costs are PER SIDE — each transaction (entry or exit) incurs the specified
fraction. A round-trip costs 2× the per-side rate.

Strategy logic (from walk_forward_ema_donchian.py):
  - Trend filter: close > EMA(ema_period)
  - Entry: Donchian(21) lower touched 2 bars ago (low.shift(2) <= DC_lower.shift(2))
    AND price has recovered above it (close > DC_lower)
    AND uptrend (close > EMA)
  - Fill at next open
  - Exit: ATR(20)-based initial stop (entry_price - sl_atr * ATR(20))
    OR ATR(20)-based trailing stop (highest_close - trail_atr * ATR(20))
  - Long-only

Outputs:
  outputs/cost_stress_20260617.csv   — detailed per-candidate, per-cost rate results
  outputs/cost_stress_20260617.txt   — plain-text summary report
"""

import sys, os, math, warnings, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pathlib import Path
import numpy as np; np.random.seed(42)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
warnings.filterwarnings('ignore')

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from trading_research_agent.config import get_output_path

# ── Candidate slate ─────────────────────────────────────────────
CANDIDATES = [
    ("XAUUSD", "GC=F", "1d", 150, 1.0, 3.0),
    ("XAGUSD", "SI=F", "4h", 50, 1.0, 4.0),
    ("STOXX50E", "^STOXX50E", "2h", 150, 1.0, 4.0),
    ("SPA35", "^IBEX", "4h", 100, 1.0, 3.0),
    ("NDX", "^NDX", "1h", 150, 1.0, 3.0),
    ("GDAXI", "^GDAXI", "1h", 50, 1.0, 4.0),
    ("J225", "^N225", "1h", 100, 1.0, 3.0),
    ("WS30", "^DJI", "1d", 150, 1.0, 3.0),
]

COST_RATES = [0.0, 0.0002, 0.0005, 0.0010, 0.0020]
DONCHIAN_WINDOW = 21
ATR_WINDOW = 20

# ── Indicators ─────────────────────────────────────────────────

def atr_vals(high, low, close, period=20):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def donchian_lower(low, period=21):
    return low.rolling(period).min()

def ema_vals(close, period):
    return close.ewm(span=period, adjust=False).mean()

# ── Strategy (matches walk_forward_ema_donchian.py) ────────────

def simulate_strategy(close, high, low, open_price, ema_period,
                      sl_atr_mult, trail_atr_mult, cost_rate):
    """Simulate the strategy and return trade list + metrics.

    Entry: close > EMA AND close > DC_lower AND low.shift(2) <= DC_lower.shift(2)
    Fill: next open (shift entry by 1 bar)
    Exit: ATR stop (low hits stop) OR trailing stop (low hits trail)
    cost_rate: fraction per side (entry and exit both incur cost)
    """
    atr_v = atr_vals(high, low, close, ATR_WINDOW)
    ema_v = ema_vals(close, ema_period)
    dc_low = donchian_lower(low, DONCHIAN_WINDOW)

    # Trend filter
    uptrend = close > ema_v

    # Donchian touch recovery entry: price touched lower band 2 bars ago,
    # now recovered above it
    touch_2ago = low.shift(2) <= dc_low.shift(2)
    recovered = close > dc_low
    raw_entry = uptrend & recovered & touch_2ago

    # Fill at next open -> shift entry by 1
    entry_signal = raw_entry.shift(1).fillna(False)

    trades = []
    in_pos = False
    entry_px = 0.0
    entry_bar = 0
    hi_since = 0.0
    trail_stop = 0.0
    hard_stop = 0.0
    cur_atr = 0.0

    n = len(close)
    close_a = close.values
    high_a = high.values
    low_a = low.values
    open_a = open_price.values
    atr_a = atr_v.values
    entry_a = entry_signal.values

    for i in range(n):
        if not in_pos:
            if entry_a[i]:
                if i + 1 < n:
                    fi = i + 1
                    fp = open_a[fi]
                    ca = atr_a[fi]
                else:
                    fi = i
                    fp = close_a[i]
                    ca = atr_a[i]
                if pd.isna(fp) or pd.isna(ca) or ca <= 0:
                    continue
                entry_px = fp
                entry_bar = fi
                hi_since = fp
                cur_atr = ca
                trail_stop = fp - trail_atr_mult * ca
                hard_stop = fp - sl_atr_mult * ca
                in_pos = True
        else:
            # Update trailing high and stop
            if i >= entry_bar:
                if close_a[i] > hi_since:
                    hi_since = close_a[i]
                    if not pd.isna(atr_a[i]) and atr_a[i] > 0:
                        cur_atr = atr_a[i]
                    trail_stop = max(trail_stop, hi_since - trail_atr_mult * cur_atr)

            exit_px = None
            etype = None

            # Hard stop: low hits the stop level
            if low_a[i] <= hard_stop:
                exit_px = min(open_a[i], hard_stop) if not pd.isna(open_a[i]) else hard_stop
                etype = "stop_loss"
            elif low_a[i] <= trail_stop and i > entry_bar:
                if i + 1 < n:
                    exit_px = open_a[i + 1]
                else:
                    exit_px = close_a[i]
                etype = "trail_stop"

            if exit_px is not None and exit_px > 0:
                total_cost = entry_px * cost_rate + exit_px * cost_rate
                net_ret = (exit_px - entry_px - total_cost) / entry_px * 100
                gross_ret = (exit_px - entry_px) / entry_px * 100
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i if etype == 'stop_loss' else min(i + 1, n - 1),
                    'entry_price': entry_px,
                    'exit_price': exit_px,
                    'gross_return_pct': gross_ret,
                    'net_return_pct': net_ret,
                    'exit_type': etype,
                })
                in_pos = False
                entry_px = 0.0
                hi_since = 0.0
                trail_stop = 0.0
                hard_stop = 0.0
                cur_atr = 0.0

    return trades


def compute_metrics(trades, close):
    """Compute summary metrics from trade list."""
    if not trades:
        return {
            'trades': 0, 'net_return_pct': 0.0, 'gross_return_pct': 0.0,
            'win_rate': 0.0, 'avg_r': 0.0, 'total_r': 0.0,
            'max_dd_pct': 0.0, 'profit_factor': 0.0,
            'avg_hold_bars': 0.0, 'stop_losses': 0, 'trail_exits': 0,
        }

    df_t = pd.DataFrame(trades)
    net_rets = df_t['net_return_pct'].values
    gross_rets = df_t['gross_return_pct'].values

    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets <= 0]

    # Equity curve from net returns
    net_cum = (1 + net_rets / 100).cumprod()
    running_max = np.maximum.accumulate(net_cum)
    drawdowns = (net_cum - running_max) / running_max
    max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0

    total_gain = sum(wins) if len(wins) > 0 else 0.0
    total_loss = abs(sum(losses)) if len(losses) > 0 else 0.0001
    pf = total_gain / total_loss if total_loss > 0 else float('inf')

    total_r = sum(net_rets)
    avg_r = total_r / len(trades) if trades else 0.0

    avg_hold = float(df_t['exit_bar'].mean() - df_t['entry_bar'].mean())
    stop_losses = int(sum(df_t['exit_type'] == 'stop_loss'))
    trail_exits = int(sum(df_t['exit_type'] == 'trail_stop'))

    return {
        'trades': len(trades),
        'net_return_pct': float((net_cum[-1] - 1) * 100),
        'gross_return_pct': float(((1 + gross_rets / 100).cumprod()[-1] - 1) * 100),
        'win_rate': len(wins) / len(net_rets) * 100 if len(net_rets) > 0 else 0.0,
        'avg_r': avg_r,
        'total_r': total_r,
        'max_dd_pct': max_dd,
        'profit_factor': pf,
        'avg_hold_bars': avg_hold,
        'stop_losses': stop_losses,
        'trail_exits': trail_exits,
    }


# ── Data Loading ────────────────────────────────────────────────

def _flatten_yf(raw):
    if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def _download_range(ticker, start, end, interval):
    """Download data in chunks if needed to handle yfinance limits."""
    import yfinance as yf
    try:
        raw = yf.download(ticker, start=start, end=end,
                          interval=interval, progress=False,
                          auto_adjust=False, threads=False)
    except Exception:
        return pd.DataFrame()
    return raw


def load_data(symbol, ticker, tf):
    """Load OHLCV, return (close, high, low, open_price, df)."""
    import yfinance as yf

    if tf == "1d":
        raw = _download_range(ticker, "2010-01-01", "2026-06-17", "1d")
        if raw.empty or len(raw) < 300:
            raise ValueError(f"Insufficient daily data")
        raw = _flatten_yf(raw)
        raw = raw.dropna(subset=['Close']).sort_index()
        if 'Adj Close' in raw.columns:
            close = raw['Adj Close'].astype(float)
        else:
            close = raw['Close'].astype(float)
        high = raw['High'].astype(float)
        low = raw['Low'].astype(float)
        open_price = raw['Open'].astype(float)
        return close, high, low, open_price, raw

    # Intraday: try recent 1y with 60m, then fallback
    end_dt = "2026-06-17"
    start_dt = "2025-06-17"  # 1 year back

    raw = _download_range(ticker, start_dt, end_dt, "60m")
    if raw.empty or len(raw) < 300:
        # Try shorter range
        raw = _download_range(ticker, "2026-01-01", end_dt, "60m")
    if raw.empty or len(raw) < 100:
        raise ValueError(f"No intraday data for {ticker}")

    raw = _flatten_yf(raw)
    raw = raw.dropna(subset=['Close']).sort_index()

    # For daily/intervals beyond daily, check if we have enough
    if tf == "1h":
        close = raw['Close'].astype(float)
        high = raw['High'].astype(float)
        low = raw['Low'].astype(float)
        open_price = raw['Open'].astype(float)
        return close, high, low, open_price, raw

    # Resample 1h -> 2h or 4h
    rule = tf.replace('h', 'h')
    resampled = raw.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    if len(resampled) < 100:
        raise ValueError(f"Resampled {tf}: only {len(resampled)} bars")
    close = resampled['Close'].astype(float)
    high = resampled['High'].astype(float)
    low = resampled['Low'].astype(float)
    open_price = resampled['Open'].astype(float)
    return close, high, low, open_price, resampled


# ── Main ────────────────────────────────────────────────────────

def main():
    output_dir = Path(get_output_path())
    csv_path = output_dir / "cost_stress_20260617.csv"
    txt_path = output_dir / "cost_stress_20260617.txt"

    all_rows = []

    for symbol, ticker, tf, ema_val, sl_atr, trail_atr in CANDIDATES:
        print(f"\n{'='*70}")
        label = f"{symbol} {tf} EMA{ema_val} SL{sl_atr}ATR Trail{trail_atr}ATR"
        print(f"  {label}")
        print(f"{'='*70}")

        try:
            close, high, low, open_price, data = load_data(symbol, ticker, tf)
            print(f"  Data: {len(data)} bars ({data.index[0].strftime('%Y-%m-%d')} → "
                  f"{data.index[-1].strftime('%Y-%m-%d')})")

            for cost_rate in COST_RATES:
                cost_bps = int(cost_rate * 10000)
                trades = simulate_strategy(
                    close, high, low, open_price,
                    ema_period=ema_val,
                    sl_atr_mult=sl_atr,
                    trail_atr_mult=trail_atr,
                    cost_rate=cost_rate,
                )
                m = compute_metrics(trades, close)

                row = {
                    'symbol': symbol, 'ticker': ticker, 'tf': tf,
                    'ema': ema_val, 'sl_atr': sl_atr, 'trail_atr': trail_atr,
                    'bars': len(data), 'cost_bps': cost_bps,
                    'cost_per_side': cost_rate,
                    **m,
                }
                all_rows.append(row)

                status = "✓" if m['trades'] > 0 else "✗"
                print(f"  [{status}] {cost_bps:3d} bps/side  trades={m['trades']:3d}  "
                      f"net={m['net_return_pct']:+7.2f}%  gross={m['gross_return_pct']:+7.2f}%  "
                      f"win={m['win_rate']:4.0f}%  PF={m['profit_factor']:.3f}  "
                      f"DD={m['max_dd_pct']:.2f}%  hold={m['avg_hold_bars']:5.1f}  "
                      f"stop={m['stop_losses']} trail={m['trail_exits']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            for cost_rate in COST_RATES:
                all_rows.append({
                    'symbol': symbol, 'ticker': ticker, 'tf': tf,
                    'ema': ema_val, 'sl_atr': sl_atr, 'trail_atr': trail_atr,
                    'bars': 0, 'cost_bps': int(cost_rate * 10000),
                    'cost_per_side': cost_rate,
                    'trades': None, 'net_return_pct': None,
                    'gross_return_pct': None, 'win_rate': None,
                    'avg_r': None, 'total_r': None, 'max_dd_pct': None,
                    'profit_factor': None, 'avg_hold_bars': None,
                    'stop_losses': None, 'trail_exits': None,
                })
            continue

    # ── Write CSV ────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_cols = ['symbol', 'ticker', 'tf', 'ema', 'sl_atr', 'trail_atr',
                'bars', 'trades', 'gross_return_pct', 'net_return_pct',
                'win_rate', 'profit_factor', 'avg_r', 'total_r',
                'max_dd_pct', 'avg_hold_bars', 'stop_losses', 'trail_exits',
                'cost_bps', 'cost_per_side']
    if len(df) > 0:
        out = df[csv_cols].copy()
        out.to_csv(csv_path, index=False)
        print(f"\nWrote CSV: {csv_path}")

    # ── Write Report ─────────────────────────────────────────
    lines = []
    lines.append("=" * 95)
    lines.append("COST STRESS TEST REPORT — ATR Trend-Pullback Candidate Slate")
    lines.append("Date: 2026-06-17")
    lines.append("=" * 95)
    lines.append("")
    lines.append("COST CONVENTION: All costs are PER SIDE.")
    lines.append("Each entry and each exit transaction incurs the specified fraction.")
    lines.append("A round-trip (entry + exit) therefore costs 2x the per-side rate.")
    lines.append("")
    lines.append(f"Strategy: Close > EMA + Donchian({DONCHIAN_WINDOW}L) pullback touch-delayed-2 + "
                 f"ATR({ATR_WINDOW}) stop + ATR trailing exit")
    lines.append(f"Slate size: {len(CANDIDATES)} candidates")
    lines.append(f"Cost rates (bps per side): {[int(c*10000) for c in COST_RATES]}")
    lines.append("")

    # Main table
    hdr_cost = "".join(f"{int(c*10000):>8d}bps" for c in COST_RATES)
    lines.append(f"{'Candidate':<44}  {'bars':>5}  {'trd':>4}  {hdr_cost}")
    lines.append("─" * 95)

    survivors_10bps = []
    failures_list = []

    for cand in CANDIDATES:
        symbol, ticker, tf, ema_val, sl_atr, trail_atr = cand
        label = f"{symbol} {tf} EMA{ema_val} SL{sl_atr}ATR Trail{trail_atr}ATR"
        rows_for_cand = df[(df['symbol'] == symbol) & (df['tf'] == tf) &
                           (df['ema'] == ema_val) & (df['sl_atr'] == sl_atr) &
                           (df['trail_atr'] == trail_atr)]

        if rows_for_cand.empty:
            lines.append(f"{label:<44}  {'--':>5}  {'--':>4}  {'ERROR':>8}  {'ERROR':>8}  {'ERROR':>8}  {'ERROR':>8}  {'ERROR':>8}")
            for cr in COST_RATES:
                failures_list.append(f"{label}  (no data)")
            continue

        bars = rows_for_cand.iloc[0]['bars']
        trades_0 = rows_for_cand[rows_for_cand['cost_bps'] == 0]['trades'].values
        trd = int(trades_0[0]) if len(trades_0) > 0 and not pd.isna(trades_0[0]) else 0

        net_strs = []
        for cr in COST_RATES:
            cb = int(cr * 10000)
            vals = rows_for_cand[rows_for_cand['cost_bps'] == cb]['net_return_pct'].values
            if len(vals) > 0 and vals[0] is not None and not pd.isna(vals[0]):
                net_strs.append(f"{vals[0]:+8.2f}")
            else:
                net_strs.append("  ERROR  ")

        ret_str = "  ".join(net_strs)
        lines.append(f"{label:<44}  {bars:>5d}  {trd:>4d}  {ret_str}")

        # Survivorship at 10 bps per side
        row10 = rows_for_cand[rows_for_cand['cost_bps'] == 10]
        if not row10.empty:
            nr10 = row10.iloc[0]['net_return_pct']
            tr10 = row10.iloc[0]['trades']
            if nr10 is not None and not pd.isna(nr10) and nr10 > 0 and tr10 is not None and tr10 >= 5:
                survivors_10bps.append(label)
            else:
                reason = f"ret={nr10:+.2f}%, {tr10} trades" if nr10 is not None else "no data"
                failures_list.append(f"{label}  ({reason})")
        else:
            failures_list.append(f"{label}  (no data at 10bps)")

    lines.append("")
    lines.append("─" * 95)
    lines.append("SURVIVORSHIP AT 10 bps PER SIDE")
    lines.append("─" * 95)
    lines.append("Criterion: net_return > 0 AND >= 5 trades")
    lines.append(f"")
    lines.append(f"Survivors ({len(survivors_10bps)}/{len(CANDIDATES)}):")
    for s in survivors_10bps:
        lines.append(f"  ✓ {s}")
    lines.append(f"")
    lines.append(f"Failures ({len(failures_list)}/{len(CANDIDATES)}):")
    for f_item in failures_list:
        lines.append(f"  ✗ {f_item}")

    # Per-cost-level survivorship
    lines.append("")
    lines.append("─" * 95)
    lines.append("SURVIVORSHIP BY COST LEVEL")
    lines.append("─" * 95)
    for cb in [0, 2, 5, 10, 20]:
        survived = []
        failed = []
        for cand in CANDIDATES:
            symbol, ticker, tf, ema_val, sl_atr, trail_atr = cand
            row = df[(df['symbol'] == symbol) & (df['tf'] == tf) &
                     (df['ema'] == ema_val) & (df['sl_atr'] == sl_atr) &
                     (df['trail_atr'] == trail_atr) & (df['cost_bps'] == cb)]
            if not row.empty:
                r = row.iloc[0]
                cl = f"{symbol} {tf}"
                if r['net_return_pct'] is not None and not pd.isna(r['net_return_pct']) and r['net_return_pct'] > 0 and r['trades'] >= 5:
                    survived.append(cl)
                else:
                    ret_s = f"{r['net_return_pct']:+.2f}%" if r['net_return_pct'] is not None else "ERR"
                    tr_s = f"{int(r['trades'])} tr" if r['trades'] is not None else "0 tr"
                    failed.append(f"{cl} ({ret_s}, {tr_s})")
        lines.append(f"\nAt {cb} bps per side:")
        lines.append(f"  Survivors ({len(survived)}): {', '.join(survived) if survived else 'none'}")
        lines.append(f"  Non-survivors ({len(failed)}): {'; '.join(failed) if failed else 'none'}")

    # Cost erosion table
    lines.append("")
    lines.append("─" * 95)
    lines.append("COST EROSION (0bps vs 20bps)")
    lines.append("─" * 95)
    for cand in CANDIDATES:
        symbol, ticker, tf, ema_val, sl_atr, trail_atr = cand
        rows_c = df[(df['symbol'] == symbol) & (df['tf'] == tf) &
                    (df['ema'] == ema_val) & (df['sl_atr'] == sl_atr) &
                    (df['trail_atr'] == trail_atr)]
        r0 = rows_c[rows_c['cost_bps'] == 0]
        r20 = rows_c[rows_c['cost_bps'] == 20]
        if not r0.empty and not r20.empty:
            nr0 = r0.iloc[0]['net_return_pct']
            nr20 = r20.iloc[0]['net_return_pct']
            if nr0 is not None and nr20 is not None and not pd.isna(nr0) and not pd.isna(nr20):
                erosion = nr0 - nr20
                lines.append(f"  {symbol:<8} {tf:<3}: 0bps={nr0:+7.2f}% → 20bps={nr20:+7.2f}%  "
                             f"erosion={erosion:+6.2f}%  tr={int(r20.iloc[0]['trades'])}  "
                             f"PF={r20.iloc[0]['profit_factor']:.2f}")

    report_text = "\n".join(lines)
    txt_path.write_text(report_text)
    print(f"\nWrote report: {txt_path}")
    print(report_text)


if __name__ == "__main__":
    main()
