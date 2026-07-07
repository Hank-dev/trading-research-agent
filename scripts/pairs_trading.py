"""Pairs trading on correlated FX pairs — statistical arbitrage approach."""
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

def run_pf(close, entries, exits, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pf = vbt.Portfolio.from_signals(close, entries, exits, freq='D')
        s = pf.stats()
    t = pf.trades
    if len(t) == 0: return None
    w = t.winning.returns.sum() if t.winning.count() > 0 else 0
    l = abs(t.losing.returns.sum()) if t.losing.count() > 0 else 0.0001
    rets = t.returns.values; eq = pf.value(); ui = ulcer_index(eq)
    sk = float(scipy_skew(rets)) if len(rets) >= 3 else np.nan
    return {'name': name, 'trades': len(t),
        'ret': float(s['Total Return [%]']), 'bench': float(s['Benchmark Return [%]']),
        'sharpe': float(s['Sharpe Ratio']), 'max_dd': float(s['Max Drawdown [%]']),
        'win_rate': float(s['Win Rate [%]']), 'pf': w/l, 'ulcer': ui, 'skew': sk}

def pairs_trade(leg1, leg2, name1, name2, lookback=60, entry_z=2.0, exit_z=0.5):
    """Pairs trading: trade the spread between two correlated assets.
    
    - Compute rolling hedge ratio (leg1 on leg2)
    - Spread = log(leg1) - hedge_ratio * log(leg2)
    - Enter when z-score crosses ±entry_z
    - Exit when z-score crosses ±exit_z (closer to 0)
    """
    # Align series
    df = pd.DataFrame({'l1': leg1, 'l2': leg2}).dropna()
    l1, l2 = df['l1'], df['l2']
    
    # Compute rolling hedge ratio and spread
    log1, log2 = np.log(l1), np.log(l2)
    
    hr = pd.Series(np.nan, index=l1.index)
    spread = pd.Series(np.nan, index=l1.index)
    
    for i in range(lookback, len(l1)):
        x = log2.iloc[i-lookback:i].values.reshape(-1, 1)
        y_arr = log1.iloc[i-lookback:i].values
        # Manual OLS
        x_mean, y_mean = x.mean(), y_arr.mean()
        beta = ((x - x_mean) * (y_arr - y_mean)).sum() / ((x - x_mean)**2).sum()
        hr.iloc[i] = beta
        spread.iloc[i] = y_arr[-1] - beta * x[-1,0]
    
    # Z-score of spread
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std().replace(0, np.nan)
    zscore = (spread - spread_mean) / spread_std
    
    # Entries and exits
    # Long leg1/short leg2 when z < -entry_z (leg1 too cheap)
    long_entry = zscore < -entry_z
    long_exit = zscore > -exit_z
    
    # Short leg1/long leg2 when z > +entry_z (leg1 too expensive)  
    short_entry = zscore > entry_z
    short_exit = zscore < exit_z
    
    # Combined position on leg1
    entries = long_entry | short_entry
    exits = long_exit | short_exit
    
    return entries, exits, zscore, spread, hr

# ── Main ─────────────────────────────────────────────
PAIR_CONFIGS = [
    ('EURUSD=X', 'GBPUSD=X', 'EURUSD', 'GBPUSD'),
    ('EURUSD=X', 'CHF=X', 'EURUSD', 'USDCHF'),  # inverse relationship
    ('AUDUSD=X', 'NZDUSD=X', 'AUDUSD', 'NZDUSD'),  # Antipodean pairs
    ('GBPUSD=X', 'USDCHF=X', 'GBPUSD', 'USDCHF'),  # often inverse
    ('USDCAD=X', 'USDNOK=X', 'USDCAD', 'USDNOK'),  # oil-linked
]

results = []
for sym1, sym2, n1, n2 in PAIR_CONFIGS:
    print(f"Testing {n1}/{n2}...", file=sys.stderr)
    try:
        p1 = yf.download(sym1, start='2015-01-01', end='2024-12-31', auto_adjust=False, progress=False)['Close'].squeeze()
        p2 = yf.download(sym2, start='2015-01-01', end='2024-12-31', auto_adjust=False, progress=False)['Close'].squeeze()
    except Exception as e:
        print(f"  FAIL: {e}", file=sys.stderr)
        continue
    
    if p1.empty or p2.empty:
        continue
    
    for lbk in [30, 60, 90, 120]:
        for ez in [1.5, 2.0, 2.5]:
            entries, exits, z, sp, hr = pairs_trade(p1, p2, n1, n2, lookback=lbk, entry_z=ez)
            if entries.empty or entries.sum() < 5:
                continue
            
            # Trade on the first leg
            r = run_pf(p1, entries, exits, f"{n1}/{n2} lb{lbk} z{ez:.1f}")
            if r:
                results.append(r)

df = pd.DataFrame(results).sort_values('pf', ascending=False)
print(f"\n{'='*120}")
print(f"PAIRS TRADING — {len(results)} configs tested, 2015-2024")
print(f"{'='*120}\n")

# Best
print(f"🔝 Top 20 by Profit Factor:")
print(f"{'Strategy':<35} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6} {'Win%':>6} {'MDD%':>6}")
print("-" * 105)
for _, r in df.head(20).iterrows():
    print(f"{r['name']:<35} {r['trades']:>5.0f} {r['ret']:>7.1f} {r['ret']-r['bench']:>7.1f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f} "
          f"{r['win_rate']:>6.1f} {r['max_dd']:>6.1f}")

# Winners
good = df[(df['ret']-df['bench']>0)&(df['pf']>1.2)&(df['trades']>=15)]
print(f"\n🏆 Qualifying (Exc>0, PF>1.2, ≥15 trades): {len(good)}")
for _, r in good.sort_values('pf', ascending=False).iterrows():
    print(f"  {r['name']:<35} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}")

# Compare to champion
print(f"\n{'='*120}")
print("VS CHAMPIONS: SMA 5/15 (PF=1.36, 77tr) | Keltner 30/2.5 (PF=2.41, 39tr) | DualSMA 5/15/3/10 (PF=1.91, 89tr)")
champ = df[(df['ret']-df['bench']>0)&(df['pf']>1.5)&(df['trades']>=30)]
print(f"  Pairs beating PF>1.5 with ≥30 trades: {len(champ)}")
for _, r in champ.sort_values('pf', ascending=False).iterrows():
    print(f"  {r['name']:<35} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}  Ulcer={r['ulcer']:.1f}")
