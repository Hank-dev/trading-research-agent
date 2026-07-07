"""Sweep strategies across FX pairs, reporting trade count, profit factor,
Ulcer Index, skew, and all standard metrics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

from scipy.stats import skew as scipy_skew

def ulcer_index(equity_curve):
    """Ulcer Index: sqrt(mean of squared drawdowns)"""
    peak = equity_curve.cummax()
    drawdown = 100 * (equity_curve - peak) / peak
    return float(np.sqrt((drawdown ** 2).mean()))

def profit_factor(total_won, total_lost):
    return total_won / abs(total_lost) if total_lost != 0 else float('inf')

def run_single(close, entries, exits, name):
    pf = vbt.Portfolio.from_signals(close, entries, exits, freq='D')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stats = pf.stats()

    trades = pf.trades
    trade_returns = trades.returns.values

    total_return = float(stats['Total Return [%]'])
    bench_return = float(stats['Benchmark Return [%]'])
    sharpe = float(stats['Sharpe Ratio'])
    max_dd = float(stats['Max Drawdown [%]'])
    win_rate = float(stats['Win Rate [%]'])
    n_trades = len(trades)

    total_won = trades.winning.returns.sum() if trades.winning.count() > 0 else 0
    total_lost = abs(trades.losing.returns.sum()) if trades.losing.count() > 0 else 0.0001
    pf_val = profit_factor(total_won, total_lost)

    ui = ulcer_index(pf.value())

    ret_skew = float(scipy_skew(trade_returns)) if len(trade_returns) >= 3 else np.nan

    return {
        'name': name,
        'trades': n_trades,
        'return': total_return,
        'bench': bench_return,
        'excess': total_return - bench_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': pf_val,
        'ulcer_index': ui,
        'skew': ret_skew,
    }


def donchian(close, entry_w, exit_w):
    high = close.rolling(entry_w).max()
    low = close.rolling(exit_w).min()
    entries = close > high.shift(1)
    exits = close < low.shift(1)
    return entries, exits


def sma_crossover(close, fast, slow):
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    return entries, exits


def rsi_reversion(close, window, oversold=30, overbought=70):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    entries = rsi < oversold
    exits = rsi > overbought
    return entries, exits


PAIRS = {
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X',
    'EURUSD': 'EURUSD=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCHF': 'USDCHF=X',
    'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
}

def load_data(symbol):
    df = yf.download(symbol, start='2015-01-01', end='2024-12-31',
                     auto_adjust=False, progress=False)
    return df['Close'].squeeze()


def main():
    results = []

    for pair_name, symbol in PAIRS.items():
        print(f"Loading {pair_name}...", file=sys.stderr)
        try:
            close = load_data(symbol)
            if close.empty:
                print(f"  SKIP: no data", file=sys.stderr)
                continue
        except Exception as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        # --- Donchian breakouts (short windows for more trades) ---
        for ew in [3, 5, 7, 10, 14, 20]:
            entries, exits = donchian(close, ew, ew)
            r = run_single(close, entries, exits, f"{pair_name} Donchian w{ew}")
            if r['trades'] > 0:
                results.append(r)

        # --- SMA crossovers (short windows) ---
        for fast, slow in [(2,5),(3,7),(5,10),(5,15),(7,21),(10,30)]:
            entries, exits = sma_crossover(close, fast, slow)
            r = run_single(close, entries, exits, f"{pair_name} SMA {fast}/{slow}")
            if r['trades'] > 0:
                results.append(r)

        # --- RSI reversal (short windows) ---
        for w in [5, 7, 10, 14]:
            for os_level, ob_level in [(25,75), (30,70)]:
                entries, exits = rsi_reversion(close, w, os_level, ob_level)
                r = run_single(close, entries, exits, f"{pair_name} RSI w{w} {os_level}/{ob_level}")
                if r['trades'] > 0:
                    results.append(r)

    # --- Output ---
    df = pd.DataFrame(results).sort_values('profit_factor', ascending=False)

    print(f"\n{'='*130}")
    print(f"Strategy Sweep Results — {len(results)} strategies tested across {len(PAIRS)} pairs")
    print(f"2015-01-01 to 2024-12-31")
    print(f"{'='*130}\n")

    # Header
    print(f"{'Strategy':<30} {'Trades':>6} {'Return%':>8} {'Bench%':>8} {'Excess%':>8} "
          f"{'Sharpe':>7} {'MaxDD%':>7} {'Win%':>6} {'PF':>6} {'Ulcer':>7} {'Skew':>6}")
    print("-" * 130)

    for _, row in df.iterrows():
        print(f"{row['name']:<30} {row['trades']:>6.0f} {row['return']:>8.1f} {row['bench']:>8.1f} "
              f"{row['excess']:>8.1f} {row['sharpe']:>7.2f} {row['max_dd']:>7.1f} {row['win_rate']:>6.1f} "
              f"{row['profit_factor']:>6.2f} {row['ulcer_index']:>7.1f} {row['skew']:>6.2f}")

    print("-" * 130)

    # Top by profit factor
    top = df.head(15)
    print("\n🏆 Top 15 by Profit Factor:\n")
    for _, row in top.iterrows():
        print(f"  {row['name']:<30} PF={row['profit_factor']:.2f}  "
              f"Trades={row['trades']:.0f}  Sharpe={row['sharpe']:.2f}  "
              f"Return={row['return']:.1f}% vs {row['bench']:.1f}%  UI={row['ulcer_index']:.1f}")

    # Top by trade count (>50 trades)
    many = df[df['trades'] >= 50].sort_values('profit_factor', ascending=False)
    if not many.empty:
        print(f"\n🔁 Strategies with 50+ trades (sorted by PF):\n")
        for _, row in many.head(10).iterrows():
            print(f"  {row['name']:<30} Trades={row['trades']:.0f}  PF={row['profit_factor']:.2f}  "
                  f"Sharpe={row['sharpe']:.2f}  Return={row['return']:.1f}% vs {row['bench']:.1f}%")

    # Winners (positive excess)
    winners = df[df['excess'] > 0].sort_values('trades', ascending=False)
    if not winners.empty:
        print(f"\n✅ Positive excess return + sorted by trade count:\n")
        for _, row in winners.head(10).iterrows():
            print(f"  {row['name']:<30} Trades={row['trades']:.0f}  Excess={row['excess']:+.1f}%  "
                  f"PF={row['profit_factor']:.2f}  Sharpe={row['sharpe']:.2f}")


if __name__ == '__main__':
    main()
