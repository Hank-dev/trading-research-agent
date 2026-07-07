"""Expanded strategy sweep — adds EMA crossover, Bollinger Band, MACD, dual Donchian."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np; np.random.seed(42)
import pandas as pd
import vectorbt as vbt
import yfinance as yf
from scipy.stats import skew as scipy_skew
import warnings

def ulcer_index(equity):
    peak = equity.cummax(); dd = 100 * (equity - peak) / peak
    return float(np.sqrt((dd ** 2).mean()))

def run(close, entries, exits, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pf = vbt.Portfolio.from_signals(close, entries, exits, freq='D')
        s = pf.stats()
    t = pf.trades
    if len(t) == 0:
        return None
    w = t.winning.returns.sum() if t.winning.count() > 0 else 0
    l = abs(t.losing.returns.sum()) if t.losing.count() > 0 else 0.0001
    rets = t.returns.values
    eq = pf.value()
    ui = ulcer_index(eq)
    sk = float(scipy_skew(rets)) if len(rets) >= 3 else np.nan
    return {
        'name': name, 'trades': len(t),
        'ret': float(s['Total Return [%]']), 'bench': float(s['Benchmark Return [%]']),
        'sharpe': float(s['Sharpe Ratio']), 'max_dd': float(s['Max Drawdown [%]']),
        'win_rate': float(s['Win Rate [%]']),
        'pf': w/l, 'ulcer': ui, 'skew': sk,
    }

# ── Strategy generators ──────────────────────────────
def donchian(close, ew):
    h = close.rolling(ew).max(); l = close.rolling(ew).min()
    return close > h.shift(1), close < l.shift(1)

def dual_donchian(close, entry_w, exit_w):
    """Separate entry and exit windows — shorter exit for tighter stops."""
    h = close.rolling(entry_w).max(); l = close.rolling(exit_w).min()
    return close > h.shift(1), close < l.shift(1)

def sma_crossover(close, fast, slow):
    sf = close.rolling(fast).mean(); ss = close.rolling(slow).mean()
    e = (sf > ss) & (sf.shift(1) <= ss.shift(1))
    x = (sf < ss) & (sf.shift(1) >= ss.shift(1))
    return e, x

def ema_crossover(close, fast, slow):
    sf = close.ewm(span=fast, adjust=False).mean()
    ss = close.ewm(span=slow, adjust=False).mean()
    e = (sf > ss) & (sf.shift(1) <= ss.shift(1))
    x = (sf < ss) & (sf.shift(1) >= ss.shift(1))
    return e, x

def rsi_reversion(close, window, os=30, ob=70):
    d = close.diff(); g = d.clip(lower=0).rolling(window).mean()
    lo = (-d.clip(upper=0)).rolling(window).mean()
    rs = g / lo.replace(0, np.nan); rsi = 100 - (100/(1+rs))
    return rsi < os, rsi > ob

def bollinger_reversion(close, window, n_std=2.0):
    """Buy when price touches lower band, sell at upper band."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = ma - n_std * std; upper = ma + n_std * std
    return close <= lower, close >= upper

def bollinger_breakout(close, window, n_std=2.0):
    """Breakout: buy above upper band, sell below lower band."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std; lower = ma - n_std * std
    e = (close > upper) & (close.shift(1) <= upper.shift(1))
    x = (close < lower) & (close.shift(1) >= lower.shift(1))
    return e, x

def macd_crossover(close, fast=12, slow=26, signal=9):
    """Classic MACD: buy when MACD crosses above signal, sell below."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    e = (macd_line > sig) & (macd_line.shift(1) <= sig.shift(1))
    x = (macd_line < sig) & (macd_line.shift(1) >= sig.shift(1))
    return e, x

# ── Main sweep ───────────────────────────────────────
PAIRS = {
    'USDJPY': 'USDJPY=X', 'GBPUSD': 'GBPUSD=X', 'EURUSD': 'EURUSD=X',
    'AUDUSD': 'AUDUSD=X', 'USDCHF': 'USDCHF=X', 'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
}

results = []
for pn, sym in PAIRS.items():
    print(f"Loading {pn}...", file=sys.stderr)
    close = yf.download(sym, start='2015-01-01', end='2024-12-31',
                        auto_adjust=False, progress=False)['Close'].squeeze()
    if close.empty: continue

    # ── Dual Donchian (entry_w ≠ exit_w) ──
    for ew, xw in [(10,5),(10,7),(15,7),(15,10),(20,10),(20,15)]:
        e, x = dual_donchian(close, ew, xw)
        r = run(close, e, x, f"{pn} dDonch {ew}/{xw}")
        if r: results.append(r)

    # ── EMA crossover ──
    for f, s in [(3,10),(5,15),(5,20),(7,21),(10,30),(12,36)]:
        e, x = ema_crossover(close, f, s)
        r = run(close, e, x, f"{pn} EMA {f}/{s}")
        if r: results.append(r)

    # ── Bollinger reversion ──
    for w in [14, 20, 28]:
        for ns in [2.0, 2.5]:
            e, x = bollinger_reversion(close, w, ns)
            r = run(close, e, x, f"{pn} BBrev w{w}s{ns:.1f}")
            if r: results.append(r)

    # ── Bollinger breakout ──
    for w in [14, 20, 28]:
        for ns in [2.0, 2.5]:
            e, x = bollinger_breakout(close, w, ns)
            r = run(close, e, x, f"{pn} BBbrk w{w}s{ns:.1f}")
            if r: results.append(r)

    # ── MACD variants ──
    for f,s,sig in [(8,21,5),(12,26,9),(6,19,5),(10,30,9)]:
        e, x = macd_crossover(close, f, s, sig)
        r = run(close, e, x, f"{pn} MACD {f}/{s}/{sig}")
        if r: results.append(r)

# ── Output ────────────────────────────────────────────
df = pd.DataFrame(results).sort_values('pf', ascending=False)
print(f"\n{'='*130}")
print(f"EXPANDED SWEEP — {len(results)} strategies, 7 pairs, 2015-2024")
print(f"{'='*130}\n")
print(f"{'Strategy':<30} {'Tr':>4} {'Ret%':>7} {'Bench%':>7} {'Exc%':>7} "
      f"{'Shrp':>6} {'PF':>6} {'Ulcr':>6} {'MDD%':>6} {'Win%':>6} {'Skew':>6}")
print("-" * 110)

for _, r in df.iterrows():
    print(f"{r['name']:<30} {r['trades']:>4.0f} {r['ret']:>7.1f} {r['bench']:>7.1f} "
          f"{r['ret']-r['bench']:>7.1f} {r['sharpe']:>6.2f} {r['pf']:>6.2f} "
          f"{r['ulcer']:>6.1f} {r['max_dd']:>6.1f} {r['win_rate']:>6.1f} {r['skew']:>6.2f}")

print("-" * 110)

# Winners (positive excess, PF>1.2, trades>=30)
good = df[(df['ret']-df['bench']>0)&(df['pf']>1.2)&(df['trades']>=30)].sort_values('pf',ascending=False)
if not good.empty:
    print(f"\n🏆 Positive excess + PF>1.2 + ≥30 trades ({len(good)} strategies):")
    for _, r in good.iterrows():
        print(f"  {r['name']:<30} Tr={r['trades']:.0f} PF={r['pf']:.2f} "
              f"Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}  "
              f"Ulcer={r['ulcer']:.1f}  Skew={r['skew']:+.2f}")

# Top 15 by PF overall
print(f"\n🔝 Top 15 by Profit Factor:")
for _, r in df.head(15).iterrows():
    print(f"  {r['name']:<30} PF={r['pf']:.2f}  Tr={r['trades']:.0f}  "
          f"Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}")

# Top 15 by trade count with PF>1
many = df[(df['trades']>=60)&(df['pf']>1)].sort_values('trades',ascending=False)
if not many.empty:
    print(f"\n📊 60+ trades + PF>1 ({len(many)} strategies):")
    for _, r in many.head(15).iterrows():
        print(f"  {r['name']:<30} Tr={r['trades']:.0f} PF={r['pf']:.2f} "
              f"Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}")
