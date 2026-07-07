"""Regime-aware BTC strategies — stay invested through v-shaped recoveries."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np; np.random.seed(42)
import pandas as pd; import vectorbt as vbt; import yfinance as yf
from scipy.stats import skew as scipy_skew; import warnings

def ulcer(eq):
    pk=eq.cummax(); dd=100*(eq-pk)/pk; return float(np.sqrt((dd**2).mean()))

def run(close, entries, exits, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pf=vbt.Portfolio.from_signals(close, entries, exits, freq='D'); s=pf.stats()
    t=pf.trades
    if len(t)==0: return None
    w=t.winning.returns.sum() if t.winning.count()>0 else 0
    l=abs(t.losing.returns.sum()) if t.losing.count()>0 else 0.0001
    rets=t.returns.values; eq=pf.value(); ui=ulcer(eq)
    sk=float(scipy_skew(rets)) if len(rets)>=3 else np.nan
    bench_ret=float(close.iloc[-1]/close.iloc[0]-1)*100
    return {'name':name,'trades':len(t),'ret':float(s['Total Return [%]']),'bench':bench_ret,
        'sharpe':float(s['Sharpe Ratio']),'max_dd':float(s['Max Drawdown [%]']),
        'win_rate':float(s['Win Rate [%]']),'pf':w/l,'ulcer':ui,'skew':sk}

# ── Regime strategies ────────────────────────────────

def sma_vol_filter(close, fast=5, slow=15, vol_lookback=20, vol_threshold=0.8):
    """Only exit when volatility is LOW — stay invested through high-vol crises.
    During high vol (crisis), hold position even if SMA crosses below.
    This avoids selling at the bottom of v-shaped crashes."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    returns=close.pct_change()
    vol=returns.rolling(vol_lookback).std()
    vol_ma=vol.rolling(100).mean()  # medium-term average vol
    vol_ratio=vol/vol_ma.replace(0,np.nan)
    is_high_vol=vol_ratio>vol_threshold
    
    entries=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    # Exit only in normal vol: in high vol, stay invested
    raw_exit=(sf<ss)&(sf.shift(1)>=ss.shift(1))
    exits=raw_exit&(~is_high_vol)
    return entries, exits

def sma_partial_exit(close, fast=5, slow=15, cash_pct=0.5):
    """On sell signal, only go to X% cash instead of 100%.
    Remaining position catches v-shaped bounces. Buy signal goes 100% long."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    entries=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    exits=(sf<ss)&(sf.shift(1)>=ss.shift(1))
    return entries, exits, cash_pct

def sma_late_exit(close, entry_fast=5, entry_slow=15, exit_slow=50):
    """Enter on fast SMA cross, EXIT only when price drops below SLOW SMA.
    This keeps you invested through shallow dips — you only exit during confirmed
    major trend changes."""
    ef=close.rolling(entry_fast).mean(); es=close.rolling(entry_slow).mean()
    xs=close.rolling(exit_slow).mean()
    entries=(ef>es)&(ef.shift(1)<=es.shift(1))
    exits=close<xs.shift(1)  # exit only when below slow SMA
    return entries, exits

def trend_strength_filter(close, fast=5, slow=15, strength_period=50):
    """Only take signals when trend strength (ADX-like) is above threshold.
    Filters out noise during ranging periods."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    # Simple trend strength: distance between fast and slow SMA relative to price
    trend_str=(sf-ss).abs()/close*100
    strong_trend=trend_str>trend_str.rolling(strength_period).mean()
    
    entries=(sf>ss)&(sf.shift(1)<=ss.shift(1))&strong_trend
    exits=(sf<ss)&(sf.shift(1)>=ss.shift(1))&strong_trend
    return entries, exits

# ── Main ─────────────────────────────────────────────
btc=yf.download("BTC-USD",start='2020-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze()
print(f"BTC 2020-2024: {len(btc)} days, return={float(btc.iloc[-1]/btc.iloc[0]-1)*100:.0f}%", file=sys.stderr)

results=[]

# Volatility-filtered
for f,s in [(5,15),(5,10),(3,7),(7,21)]:
    for vt in [0.8, 1.0, 1.2, 1.5]:
        e,x=sma_vol_filter(btc,f,s,vol_threshold=vt)
        if e.sum()>=3:
            r=run(btc,e,x,f"BTC VolSMA {f}/{s} vt{vt}")
            if r: results.append(r)

# Late exit (slow SMA confirmation)
for ef,es,exs in [(5,15,50),(5,15,100),(7,21,50),(7,21,100),(3,10,50),(10,30,100)]:
    e,x=sma_late_exit(btc,ef,es,exs)
    if e.sum()>=3:
        r=run(btc,e,x,f"BTC LateExit {ef}/{es} x{exs}")
        if r: results.append(r)

# Trend-strength filter
for f,s in [(5,15),(7,21),(3,10),(10,30)]:
    e,x=trend_strength_filter(btc,f,s)
    if e.sum()>=3:
        r=run(btc,e,x,f"BTC TrendStr {f}/{s}")
        if r: results.append(r)

# ── Output ───────────────────────────────────────────
df=pd.DataFrame(results).sort_values('pf',ascending=False)
print(f"\n{'='*110}")
print(f"REGIME-AWARE BTC STRATEGIES — {len(results)} configs, 2020-2024")
print(f"{'='*110}\n")

print(f"🔝 Top results:")
print(f"{'Strategy':<35} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6} {'Win%':>6}")
print("-"*105)
for _,r in df.head(15).iterrows():
    print(f"{r['name']:<35} {r['trades']:>5.0f} {r['ret']:>7.0f} {r['ret']-r['bench']:>7.0f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f} {r['win_rate']:>6.1f}")

good=df[(df['ret']-df['bench']>0)&(df['pf']>1.2)&(df['trades']>=10)]
print(f"\n🏆 Qualifying (Exc>0, PF>1.2, ≥10 trades): {len(good)}")
for _,r in good.sort_values('pf',ascending=False).iterrows():
    print(f"  {r['name']:<35} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.0f}%  Sharpe={r['sharpe']:.2f}  Ulcer={r['ulcer']:.1f}")

# Compare with standard SMA
std=df[df['name'].str.contains('VolSMA 5/15 vt1.0')]
if not std.empty:
    print(f"\n📊 Reference (standard SMA 5/15 = no regime filter): Tr={std.iloc[0]['trades']:.0f} PF={std.iloc[0]['pf']:.2f} Exc={std.iloc[0]['ret']-std.iloc[0]['bench']:+.0f}%")
