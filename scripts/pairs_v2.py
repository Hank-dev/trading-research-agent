"""Pairs trading — clean implementation with proper index alignment."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np; np.random.seed(42)
import pandas as pd; import vectorbt as vbt; import yfinance as yf
from scipy.stats import skew as scipy_skew; import warnings

def ulcer(eq):
    pk=eq.cummax(); dd=100*(eq-pk)/pk; return float(np.sqrt((dd**2).mean()))

def run_pf(close, entries, exits, name):
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

def spread_signals(leg1, leg2, lookback=60, entry_z=2.0, exit_z=0.5):
    """Trade the log spread between two pairs."""
    df=pd.DataFrame({'l1':leg1,'l2':leg2}).dropna()
    x=np.log(df['l1']); y=np.log(df['l2'])
    lr=x-y
    ma=lr.rolling(lookback).mean(); st=lr.rolling(lookback).std().replace(0,np.nan)
    z=(lr-ma)/st
    entries=pd.Series(False,index=df.index); exits=pd.Series(False,index=df.index)
    # State machine: -1=short, 0=neutral, 1=long
    state=pd.Series(0,index=df.index)
    pos=0
    for i in range(1,len(z)):
        if pos==0:
            if z.iloc[i]<-entry_z: pos=1; entries.iloc[i]=True  # long leg1
            elif z.iloc[i]>entry_z: pos=-1; entries.iloc[i]=True  # short leg1
        elif pos==1 and z.iloc[i]>-exit_z: pos=0; exits.iloc[i]=True
        elif pos==-1 and z.iloc[i]<exit_z: pos=0; exits.iloc[i]=True
        state.iloc[i]=pos
    return entries, exits, z

# Also try: spread as simple rolling z-score (OR rule: no state machine)
def spread_signals_or(leg1, leg2, lookback=60, entry_z=2.0, exit_z=0.0):
    """Simpler: enter on any extreme, exit on crossing zero — more trades."""
    df=pd.DataFrame({'l1':leg1,'l2':leg2}).dropna()
    x=np.log(df['l1']); y=np.log(df['l2'])
    lr=x-y; ma=lr.rolling(lookback).mean(); st=lr.rolling(lookback).std().replace(0,np.nan)
    z=(lr-ma)/st
    entries=z.abs()>entry_z
    exits=(z.abs()<exit_z)|(np.sign(z)!=np.sign(z.shift(1).fillna(0)))
    return entries, exits, z

# ── Main ─────────────────────────────────────────────
PAIRS=[('EURUSD=X','GBPUSD=X','EUR/GBP'),('AUDUSD=X','NZDUSD=X','AUD/NZD'),
       ('EURUSD=X','CHF=X','EUR/CHF'),('GBPUSD=X','USDCHF=X','GBP/CHF')]
results=[]

for s1,s2,nm in PAIRS:
    print(f"Loading {nm}...", file=sys.stderr)
    p1=yf.download(s1,start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze()
    p2=yf.download(s2,start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze()
    if p1.empty or p2.empty: continue

    for lb in [30,60,90,120]:
        for ez in [1.5,2.0,2.5]:
            e,x,z=spread_signals(p1,p2,lb,ez)
            if e.sum()>=5:
                r=run_pf(p1,e,x,f"{nm} sm lb{lb} z{ez:.1f}")
                if r: results.append(r)

            e,x,z=spread_signals_or(p1,p2,lb,ez)
            if e.sum()>=5:
                r=run_pf(p1,e,x,f"{nm} or lb{lb} z{ez:.1f}")
                if r: results.append(r)

df=pd.DataFrame(results).sort_values('pf',ascending=False)
print(f"\n{'='*110}")
print(f"PAIRS TRADING — {len(results)} configs, 2015-2024")
print(f"{'='*110}\n")
print(f"{'Strategy':<35} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6} {'Win%':>6}")
print("-"*100)
for _,r in df.head(15).iterrows():
    print(f"{r['name']:<35} {r['trades']:>5.0f} {r['ret']:>7.1f} {r['ret']-r['bench']:>7.1f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f} {r['win_rate']:>6.1f}")

good=df[(df['ret']-df['bench']>0)&(df['pf']>1.15)&(df['trades']>=20)]
print(f"\n🏆 Qualifying (Exc>0, PF>1.15, ≥20 trades): {len(good)}")
for _,r in good.sort_values('pf',ascending=False).iterrows():
    print(f"  {r['name']:<35} Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}%  Sharpe={r['sharpe']:.2f}  Ulcer={r['ulcer']:.1f}")
