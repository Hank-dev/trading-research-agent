"""New strategies: ATR trailing stops, SuperTrend, multi-TF, RSI filter on FX pairs."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np; np.random.seed(42)
import pandas as pd; import vectorbt as vbt; import yfinance as yf
from scipy.stats import skew; import warnings

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
    sk=float(skew(rets)) if len(rets)>=3 else np.nan
    return {'name':name,'trades':len(t),'ret':float(s['Total Return [%]']),
        'bench':float(s['Benchmark Return [%]']),'sharpe':float(s['Sharpe Ratio']),
        'max_dd':float(s['Max Drawdown [%]']),'win_rate':float(s['Win Rate [%]']),
        'pf':w/l,'ulcer':ui,'skew':sk}

# ── Strategy builders ────────────────────────────────

def sma_atr_trail(close, fast=5, slow=15, atr_mult=2.0, atr_p=14):
    """SMA crossover entry, ATR trailing stop exit."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    hi=close.rolling(2).max(); lo=close.rolling(2).min()
    tr=pd.concat([hi-lo,(hi-close.shift(1)).abs(),(lo-close.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(atr_p).mean()
    entries=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    # Trailing stop
    exits=pd.Series(False,index=close.index)
    pos=False; highest=0; stop=0
    for i in range(len(close)):
        if entries.iloc[i]:
            pos=True; highest=close.iloc[i]; stop=highest-atr_mult*atr.iloc[i]
        elif pos:
            if close.iloc[i]>highest:
                highest=close.iloc[i]; stop=highest-atr_mult*atr.iloc[i]
            if close.iloc[i]<=stop:
                exits.iloc[i]=True; pos=False
    return entries, exits

def supertrend(close, period=10, mult=3.0):
    """SuperTrend: ATR-based trend following with stop-and-reverse."""
    hi=close.rolling(2).max(); lo=close.rolling(2).min()
    tr=pd.concat([hi-lo,(hi-close.shift(1)).abs(),(lo-close.shift(1)).abs()],axis=1).max(axis=1)
    atr=tr.rolling(period).mean()
    hl2=(hi+lo)/2
    upper=hl2+mult*atr; lower=hl2-mult*atr
    
    entries=pd.Series(False,index=close.index); exits=pd.Series(False,index=close.index)
    pos=False; up=upper.copy(); dn=lower.copy()
    for i in range(1,len(close)):
        up.iloc[i]=min(up.iloc[i],up.iloc[i-1]) if close.iloc[i-1]<=up.iloc[i-1] else up.iloc[i]
        dn.iloc[i]=max(dn.iloc[i],dn.iloc[i-1]) if close.iloc[i-1]>=dn.iloc[i-1] else dn.iloc[i]
        if close.iloc[i]>up.iloc[i-1] and not pos:
            entries.iloc[i]=True; pos=True
        elif close.iloc[i]<dn.iloc[i-1] and pos:
            exits.iloc[i]=True; pos=False
    return entries, exits

def sma_multi_tf(close, fast=5, slow=15, filter_fast=20, filter_slow=50):
    """SMA 5/15 signals filtered by 20/50 trend direction."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    ff=close.rolling(filter_fast).mean(); fs=close.rolling(filter_slow).mean()
    uptrend=ff>fs  # only trade in direction of medium-term trend
    raw_entry=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    raw_exit=(sf<ss)&(sf.shift(1)>=ss.shift(1))
    entries=raw_entry&uptrend  # only go long when medium trend is up
    exits=raw_exit&(~uptrend)  # only exit when medium trend is down
    return entries, exits

def sma_rsi_filter(close, fast=5, slow=15, rsi_w=14, rsi_entry_max=70, rsi_exit_min=30):
    """SMA crossover with RSI filter: don't buy overbought, don't sell oversold."""
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    d=close.diff(); g=d.clip(lower=0).rolling(rsi_w).mean()
    lo=(-d.clip(upper=0)).rolling(rsi_w).mean()
    rs=g/lo.replace(0,np.nan); rsi=100-(100/(1+rs))
    raw_entry=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    raw_exit=(sf<ss)&(sf.shift(1)>=ss.shift(1))
    entries=raw_entry&(rsi<rsi_entry_max)  # don't buy when already overbought
    exits=raw_exit&(rsi>rsi_exit_min)      # don't sell when oversold
    return entries, exits

# ── Main ─────────────────────────────────────────────
PAIRS={'USDJPY':'USDJPY=X','AUDUSD':'AUDUSD=X','EURUSD':'EURUSD=X','GBPUSD':'GBPUSD=X'}
PERIODS={'2015-24':('2015-01-01','2024-12-31'),'2020-24':('2020-01-01','2024-12-31')}
results=[]

for pname,(start,end) in PERIODS.items():
    for name,sym in PAIRS.items():
        close=yf.download(sym,start=start,end=end,auto_adjust=False,progress=False)['Close'].squeeze()
        if close.empty: continue
        prefix=f"{name} {pname}"

        # ATR trailing — aggressive and conservative
        for m in [1.5,2.0,2.5,3.0]:
            e,x=sma_atr_trail(close,5,15,m); r=run(close,e,x,f"{prefix} ATRtrail{m:.1f}")
            if r: results.append(r)

        # SuperTrend
        for p,m in [(7,3),(10,3),(14,3),(10,2.5),(14,2.5)]:
            e,x=supertrend(close,p,m); r=run(close,e,x,f"{prefix} SuperT{p}/{m}")
            if r: results.append(r)

        # Multi-TF
        for ff,fs in [(20,50),(50,100),(30,70)]:
            e,x=sma_multi_tf(close,5,15,ff,fs); r=run(close,e,x,f"{prefix} MultiTF{ff}/{fs}")
            if r: results.append(r)

        # RSI-filtered
        for rem in [65,70,75]:
            e,x=sma_rsi_filter(close,5,15,14,rem,30); r=run(close,e,x,f"{prefix} RSIlt{rem}")
            if r: results.append(r)

# ── Output ───────────────────────────────────────────
df=pd.DataFrame(results).sort_values('pf',ascending=False)
print(f"\n{'='*120}")
print(f"NEW STRATEGIES — {len(results)} tests across {len(PAIRS)} pairs, 2 periods")
print(f"{'='*120}\n")

good=df[(df['ret']-df['bench']>0)&(df['pf']>1.3)&(df['trades']>=10)]
print(f"🏆 Qualifying (Exc>0, PF>1.3, ≥10 trades): {len(good)}")
print(f"{'Strategy':<38} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Skew':>6}")
print("-"*110)
for _,r in good.sort_values('pf',ascending=False).iterrows():
    print(f"{r['name']:<38} {r['trades']:>5.0f} {r['ret']:>7.0f} {r['ret']-r['bench']:>7.0f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['skew']:>6.2f}")

# Compare ATR trail to SMA 5/15 baseline by pair
print(f"\n📊 ATR trailing vs SMA 5/15 baseline:")
for name in PAIRS:
    sma_base=df[(df['name'].str.contains(name))&(df['name'].str.contains('ATRtrail2.0'))]
    if not sma_base.empty:
        r=sma_base.iloc[0]
        print(f"  {name:<10} ATR2.0: Tr={r['trades']:.0f} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.0f}% Ulcer={r['ulcer']:.1f}")
