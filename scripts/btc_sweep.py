"""Full strategy sweep on BTC-USD — crypto-optimized parameters."""
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
    return {'name':name,'trades':len(t),'ret':float(s['Total Return [%]']),
        'bench':float(s['Benchmark Return [%]']),'sharpe':float(s['Sharpe Ratio']),
        'max_dd':float(s['Max Drawdown [%]']),'win_rate':float(s['Win Rate [%]']),
        'pf':w/l,'ulcer':ui,'skew':sk}

# === Strategy builders ===
def donchian(close, ew):
    h=close.rolling(ew).max(); l=close.rolling(ew).min()
    return close>h.shift(1), close<l.shift(1)

def dual_donchian(close, ew, xw):
    h=close.rolling(ew).max(); l=close.rolling(xw).min()
    return close>h.shift(1), close<l.shift(1)

def sma_cross(close, f, s):
    sf=close.rolling(f).mean(); ss=close.rolling(s).mean()
    return (sf>ss)&(sf.shift(1)<=ss.shift(1)), (sf<ss)&(sf.shift(1)>=ss.shift(1))

def dual_sma(close, ef, es, xf, xs):
    efm=close.rolling(ef).mean(); esm=close.rolling(es).mean()
    xfm=close.rolling(xf).mean(); xsm=close.rolling(xs).mean()
    return (efm>esm)&(efm.shift(1)<=esm.shift(1)), (xfm<xsm)&(xfm.shift(1)>=xsm.shift(1))

def ema_cross(close, f, s):
    sf=close.ewm(span=f,adjust=False).mean(); ss=close.ewm(span=s,adjust=False).mean()
    return (sf>ss)&(sf.shift(1)<=ss.shift(1)), (sf<ss)&(sf.shift(1)>=ss.shift(1))

def rsi_rev(close, w, os=30, ob=70):
    d=close.diff(); g=d.clip(lower=0).rolling(w).mean()
    lo=(-d.clip(upper=0)).rolling(w).mean()
    rs=g/lo.replace(0,np.nan); rsi=100-(100/(1+rs))
    return rsi<os, rsi>ob

def bollinger_rev(close, w, ns=2.0):
    ma=close.rolling(w).mean(); std=close.rolling(w).std()
    return close<=ma-ns*std, close>=ma+ns*std

def bollinger_brk(close, w, ns=2.0):
    ma=close.rolling(w).mean(); std=close.rolling(w).std()
    u=ma+ns*std; l=ma-ns*std
    return (close>u)&(close.shift(1)<=u.shift(1)), (close<l)&(close.shift(1)>=l.shift(1))

def macd_cross(close, f=12, s=26, sig=9):
    ef=close.ewm(span=f,adjust=False).mean(); es=close.ewm(span=s,adjust=False).mean()
    ml=ef-es; sl=ml.ewm(span=sig,adjust=False).mean()
    return (ml>sl)&(ml.shift(1)<=sl.shift(1)), (ml<sl)&(ml.shift(1)>=sl.shift(1))

def keltner(close, ep=20, atr_p=10, am=2.0):
    hi=close.rolling(2).max(); lo=close.rolling(2).min()
    tr=pd.concat([hi-lo,(hi-close.shift(1)).abs(),(lo-close.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(atr_p).mean(); ema=close.ewm(span=ep,adjust=False).mean()
    return close<ema-am*atr, close>ema+am*atr

# === MAIN ===
print("Loading BTC-USD...", file=sys.stderr)
btc=yf.download("BTC-USD",start='2020-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze()
if btc.empty: print("NO DATA"); sys.exit(1)
print(f"BTC range: {btc.index[0].date()} to {btc.index[-1].date()}, {len(btc)} days", file=sys.stderr)

results=[]

# Donchian — crypto-friendly windows
for ew in [3,5,7,10,14,20,30,50,70]:
    e,x=donchian(btc,ew); r=run(btc,e,x,f"BTC Donch w{ew}")
    if r: results.append(r)

# Dual Donchian
for ew,xw in [(7,3),(10,5),(14,7),(20,10),(30,14),(50,20),(70,30)]:
    e,x=dual_donchian(btc,ew,xw); r=run(btc,e,x,f"BTC dDonch {ew}/{xw}")
    if r: results.append(r)

# SMA crossover — short windows for crypto
for f,s in [(3,7),(5,10),(5,15),(7,21),(10,30),(3,10),(2,5),(7,14)]:
    e,x=sma_cross(btc,f,s); r=run(btc,e,x,f"BTC SMA {f}/{s}")
    if r: results.append(r)

# Dual SMA
for ef,es,xf,xs in [(5,15,3,10),(3,10,5,15),(5,10,7,15),(7,21,10,30)]:
    e,x=dual_sma(btc,ef,es,xf,xs); r=run(btc,e,x,f"BTC DualSMA {ef}/{es}/{xf}/{xs}")
    if r: results.append(r)

# EMA crossover
for f,s in [(3,10),(5,15),(7,21),(10,30),(5,20)]:
    e,x=ema_cross(btc,f,s); r=run(btc,e,x,f"BTC EMA {f}/{s}")
    if r: results.append(r)

# RSI mean reversion
for w in [5,7,10,14,21]:
    for os,ob in [(25,75),(30,70)]:
        e,x=rsi_rev(btc,w,os,ob); r=run(btc,e,x,f"BTC RSI w{w} {os}/{ob}")
        if r: results.append(r)

# Bollinger reversion + breakout
for w in [14,20,28]:
    for ns in [2.0,2.5]:
        e,x=bollinger_rev(btc,w,ns); r=run(btc,e,x,f"BTC BBrev w{w}s{ns:.1f}")
        if r: results.append(r)
        e,x=bollinger_brk(btc,w,ns); r=run(btc,e,x,f"BTC BBbrk w{w}s{ns:.1f}")
        if r: results.append(r)

# MACD
for f,s,sig in [(6,19,5),(8,21,5),(12,26,9),(10,30,9)]:
    e,x=macd_cross(btc,f,s,sig); r=run(btc,e,x,f"BTC MACD {f}/{s}/{sig}")
    if r: results.append(r)

# Keltner
for ep in [15,20,30]:
    for am in [1.5,2.0,2.5]:
        e,x=keltner(btc,ep,10,am); r=run(btc,e,x,f"BTC Keltner {ep}/{am}")
        if r: results.append(r)

# === OUTPUT ===
df=pd.DataFrame(results).sort_values('pf',ascending=False)
bench_ret=float(btc.iloc[-1]/btc.iloc[0]-1)*100
print(f"\n{'='*120}")
print(f"BTC-USD FULL SWEEP — {len(results)} strategies, 2020-2024 (bench: {bench_ret:.0f}%)")
print(f"{'='*120}\n")

# TOP 20 by PF
print(f"🔝 Top 20 by Profit Factor:")
print(f"{'Strategy':<32} {'Tr':>5} {'Ret%':>8} {'Exc%':>8} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6} {'Win%':>6} {'MDD%':>7}")
print("-"*110)
for _,r in df.head(20).iterrows():
    print(f"{r['name']:<32} {r['trades']:>5.0f} {r['ret']:>8.0f} {r['ret']-r['bench']:>8.0f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f} "
          f"{r['win_rate']:>6.1f} {r['max_dd']:>7.1f}")

# Winners (Exc>0, PF>1.3, Tr>=20)
good=df[(df['ret']-df['bench']>0)&(df['pf']>1.3)&(df['trades']>=20)]
print(f"\n🏆 Qualifying (Exc>0, PF>1.3, ≥20 trades): {len(good)}")
for _,r in good.sort_values('pf',ascending=False).iterrows():
    print(f"  {r['name']:<32} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.0f}%  Sharpe={r['sharpe']:.2f}  Ulcer={r['ulcer']:.1f}  Skew={r['skew']:+.2f}")

# High freq (80+ trades, PF>1.1)
freq=df[(df['trades']>=80)&(df['pf']>1.1)]
print(f"\n🔁 80+ trades + PF>1.1: {len(freq)}")
for _,r in freq.sort_values('pf',ascending=False).iterrows():
    print(f"  {r['name']:<32} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.0f}%  Sharpe={r['sharpe']:.2f}")

# Best by category
best_donch=df[df['name'].str.contains('Donch')].head(3)
best_sma=df[df['name'].str.contains('SMA')].head(3)
best_ema=df[df['name'].str.contains('EMA')].head(3)
best_rsi=df[df['name'].str.contains('RSI')].head(3)
best_bb=df[df['name'].str.contains('BB')].head(3)
best_macd=df[df['name'].str.contains('MACD')].head(3)
best_kelt=df[df['name'].str.contains('Keltner')].head(3)

print(f"\n📊 Best per family:")
print(f"  Donchian:  {best_donch.iloc[0]['name']:<30} PF={best_donch.iloc[0]['pf']:.2f} Tr={best_donch.iloc[0]['trades']:.0f}")
print(f"  SMA:       {best_sma.iloc[0]['name']:<30} PF={best_sma.iloc[0]['pf']:.2f} Tr={best_sma.iloc[0]['trades']:.0f}")
print(f"  EMA:       {best_ema.iloc[0]['name']:<30} PF={best_ema.iloc[0]['pf']:.2f} Tr={best_ema.iloc[0]['trades']:.0f}")
print(f"  RSI:       {best_rsi.iloc[0]['name']:<30} PF={best_rsi.iloc[0]['pf']:.2f} Tr={best_rsi.iloc[0]['trades']:.0f}")
print(f"  BB/Bands:  {best_bb.iloc[0]['name']:<30} PF={best_bb.iloc[0]['pf']:.2f} Tr={best_bb.iloc[0]['trades']:.0f}")
print(f"  MACD:      {best_macd.iloc[0]['name']:<30} PF={best_macd.iloc[0]['pf']:.2f} Tr={best_macd.iloc[0]['trades']:.0f}")
print(f"  Keltner:   {best_kelt.iloc[0]['name']:<30} PF={best_kelt.iloc[0]['pf']:.2f} Tr={best_kelt.iloc[0]['trades']:.0f}")
