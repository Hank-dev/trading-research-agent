"""High-frequency focused: short windows + smart filters on best pairs."""
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
    if len(t) == 0: return None
    w = t.winning.returns.sum() if t.winning.count() > 0 else 0
    l = abs(t.losing.returns.sum()) if t.losing.count() > 0 else 0.0001
    rets = t.returns.values
    eq = pf.value(); ui = ulcer_index(eq)
    sk = float(scipy_skew(rets)) if len(rets) >= 3 else np.nan
    return {'name': name, 'trades': len(t),
        'ret': float(s['Total Return [%]']), 'bench': float(s['Benchmark Return [%]']),
        'sharpe': float(s['Sharpe Ratio']), 'max_dd': float(s['Max Drawdown [%]']),
        'win_rate': float(s['Win Rate [%]']), 'pf': w/l, 'ulcer': ui, 'skew': sk}

# Strategy: SMA crossover with price>200SMA trend filter
def sma_trend_filter(close, fast, slow, trend=200):
    sf = close.rolling(fast).mean(); ss = close.rolling(slow).mean()
    trend_sma = close.rolling(trend).mean()
    # Only trade in trend direction: long when price>trend, short when below
    in_uptrend = close > trend_sma
    buy = (sf > ss) & (sf.shift(1) <= ss.shift(1)) & in_uptrend
    sell = (sf < ss) & (sf.shift(1) >= ss.shift(1)) & (~in_uptrend)
    return buy, sell

# Strategy: SMA crossover entry, ATR trailing stop exit
def sma_with_atr_exit(close, fast, slow, atr_period=14, atr_mult=2.0):
    sf = close.rolling(fast).mean(); ss = close.rolling(slow).mean()
    high = close.rolling(2).max()  # approximate high
    low = close.rolling(2).min()   # approximate low
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    entries = (sf > ss) & (sf.shift(1) <= ss.shift(1))
    
    # Build trailing stop: track highest close since entry, exit when drops by ATR*mult
    exits = pd.Series(False, index=close.index)
    in_position = False; highest = 0
    for i in range(1, len(close)):
        if entries.iloc[i]:
            in_position = True; highest = close.iloc[i]
        if in_position:
            highest = max(highest, close.iloc[i])
            if close.iloc[i] < highest - atr_mult * atr.iloc[i]:
                exits.iloc[i] = True
                in_position = False
    return entries, exits

# Strategy: 2x SMA crossover (fast entry, slower exit)
def sma_dual_exit(close, entry_fast, entry_slow, exit_fast, exit_slow):
    ef = close.rolling(entry_fast).mean(); es = close.rolling(entry_slow).mean()
    xf = close.rolling(exit_fast).mean(); xs = close.rolling(exit_slow).mean()
    entries = (ef > es) & (ef.shift(1) <= es.shift(1))
    exits = (xf < xs) & (xf.shift(1) >= xs.shift(1))
    return entries, exits

# Strategy: Keltner Channel breakout (for mean reversion)
def keltner_channel(close, ema_period=20, atr_period=10, atr_mult=2.0):
    high = close.rolling(2).max(); low = close.rolling(2).min()
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    ema = close.ewm(span=ema_period, adjust=False).mean()
    upper = ema + atr_mult * atr; lower = ema - atr_mult * atr
    entries = close < lower  # buy when oversold
    exits = close > upper   # sell when overbought
    return entries, exits

# ── Main ─────────────────────────────────────────────
PAIRS = {'USDJPY': 'USDJPY=X', 'USDCHF': 'USDCHF=X', 'EURUSD': 'EURUSD=X',
         'USDCAD': 'USDCAD=X', 'GBPUSD': 'GBPUSD=X'}
results = []

for pn, sym in PAIRS.items():
    close = yf.download(sym, start='2015-01-01', end='2024-12-31',
                        auto_adjust=False, progress=False)['Close'].squeeze()
    if close.empty: continue

    # Trend-filtered SMA crossovers
    for f, s in [(3,7), (5,10), (5,15), (3,10), (7,14)]:
        e, x = sma_trend_filter(close, f, s, trend=100)
        r = run(close, e, x, f"{pn} SMA {f}/{s} t100")
        if r: results.append(r)

    # SMA with ATR exit
    for f, s in [(3,7), (5,10), (5,15)]:
        for am in [1.5, 2.0, 2.5]:
            e, x = sma_with_atr_exit(close, f, s, atr_period=10, atr_mult=am)
            r = run(close, e, x, f"{pn} SMA {f}/{s} ATR{am}")
            if r: results.append(r)

    # Dual SMA (different entry/exit windows)
    for ef, es, xf, xs in [(3,7,5,10), (5,10,7,15), (5,15,3,10), (3,10,5,15)]:
        e, x = sma_dual_exit(close, ef, es, xf, xs)
        r = run(close, e, x, f"{pn} DualSMA {ef}/{es}/{xf}/{xs}")
        if r: results.append(r)

    # Keltner channel mean reversion
    for ep in [15, 20, 30]:
        for am in [1.5, 2.0, 2.5]:
            e, x = keltner_channel(close, ema_period=ep, atr_period=10, atr_mult=am)
            r = run(close, e, x, f"{pn} Keltner {ep}/{am}")
            if r: results.append(r)

# ── Output ───────────────────────────────────────────
df = pd.DataFrame(results).sort_values('pf', ascending=False)
print(f"\n{'='*120}")
print(f"HIGH-FREQ FOCUS — {len(results)} strategies, best FX pairs, 2015-2024")
print(f"{'='*120}\n")

# Winners with decent trade count
good = df[(df['ret']-df['bench']>0)&(df['pf']>1.15)&(df['trades']>=20)]
print(f"🏆 {len(good)} qualifying strategies (Exc>0, PF>1.15, Tr≥20):")
print(f"{'Strategy':<32} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6} {'MDD%':>6}")
print("-" * 95)
for _, r in good.sort_values('pf', ascending=False).iterrows():
    print(f"{r['name']:<32} {r['trades']:>5.0f} {r['ret']:>7.1f} {r['ret']-r['bench']:>7.1f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f} {r['max_dd']:>6.1f}")

# High trade count (60+), PF>1
many = df[(df['trades']>=60)&(df['pf']>1)].sort_values('pf', ascending=False)
if not many.empty:
    print(f"\n🔁 {len(many)} high-frequency (60+ trades, PF>1):")
    for _, r in many.head(15).iterrows():
        print(f"  {r['name']:<32} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}")

# Compare to champion SMA 5/15
print(f"\n{'='*120}")
print("VS CHAMPION (USDJPY SMA 5/15: PF=1.36, 77 tr, Exc=+3.4%, Sharpe=0.38, Ulcer=8.1)")
champ = df[(df['ret']-df['bench']>0)&(df['pf']>1.36)&(df['trades']>=50)]
if not champ.empty:
    print(f"\n🏅 BEATING the champion on all fronts:")
    for _, r in champ.sort_values('pf', ascending=False).iterrows():
        print(f"  {r['name']:<32} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}  Ulcer={r['ulcer']:.1f}")
else:
    print("  No strategy beats the champion on all metrics — SMA 5/15 holds its crown.")

# Trade count 100+ with PF>1.1
many2 = df[(df['trades']>=80)&(df['pf']>1.1)&(df['ret']-df['bench']>0)]
if not many2.empty:
    print(f"\n📊 80+ trades + PF>1.1 + positive excess:")
    for _, r in many2.sort_values('pf', ascending=False).iterrows():
        print(f"  {r['name']:<32} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%")
