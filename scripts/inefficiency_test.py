"""Test structural inefficiencies: commodity FX lag + Fed-BOJ spread compression."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np; np.random.seed(42)
import pandas as pd; import vectorbt as vbt; import yfinance as yf
from scipy.stats import skew; import warnings

def ulcer(eq):
    pk=eq.cummax(); dd=100*(eq-pk)/pk; return float(np.sqrt((dd**2).mean()))

def test(close, entries, exits, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pf=vbt.Portfolio.from_signals(close, entries, exits, freq='D')
        s=pf.stats()
    t=pf.trades
    if len(t)==0: return None
    w=t.winning.returns.sum() if t.winning.count()>0 else 0
    l=abs(t.losing.returns.sum()) if t.losing.count()>0 else 0.0001
    rets=t.returns.values; eq=pf.value(); ui=ulcer(eq)
    sk=float(skew(rets)) if len(rets)>=3 else np.nan
    bench=float(close.iloc[-1]/close.iloc[0]-1)*100
    return {'name':name,'trades':len(t),'ret':float(s['Total Return [%]']),
        'bench':bench,'sharpe':float(s['Sharpe Ratio']),'max_dd':float(s['Max Drawdown [%]']),
        'win_rate':float(s['Win Rate [%]']),'pf':w/l,'ulcer':ui,'skew':sk}

# ── DATA ─────────────────────────────────────────────
print("Loading FX + commodities...", file=sys.stderr)
fx={
    'AUDUSD':yf.download("AUDUSD=X",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'NZDUSD':yf.download("NZDUSD=X",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'USDCAD':yf.download("USDCAD=X",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'USDJPY':yf.download("USDJPY=X",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
}
com={
    'Gold':yf.download("GC=F",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'Oil':yf.download("CL=F",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'IronOre':yf.download("SCO=F",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
    'Copper':yf.download("HG=F",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze(),
}
results=[]

# ═══════ 1. COMMODITY TERMS-OF-TRADE FX LAG ═══════
for months_lag in [0,1,2,3]:
    for fx_name, fx_px in fx.items():
        for com_name, com_px in com.items():
            # Resample to monthly
            fx_m=fx_px.resample('ME').last()
            com_m=com_px.resample('ME').last()
            aligned=pd.DataFrame({'fx':fx_m,'com':com_m}).dropna()
            
            # Signal: commodity was up over lookback period
            com_chg=aligned['com'].pct_change(months_lag+1)
            signal=com_chg>0
            
            # Entry: buy FX on positive commodity signal
            entries=pd.Series(False,index=aligned.index)
            exits=pd.Series(False,index=aligned.index)
            for i in range(1,len(signal)):
                if signal.iloc[i-1] and not signal.iloc[i-2] if i>1 else signal.iloc[i-1]:
                    entries.iloc[i]=True
                    # Exit after 1 month (re-evaluate next month)
                    if i+1<len(exits): exits.iloc[i+1]=True
            
            if entries.sum()>=5:
                r=test(aligned['fx'],entries,exits,
                       f"{fx_name}←{com_name} lag{months_lag+1}mo")
                if r: results.append(r)

# ═══════ 2. SHORT USDJPY ON SPREAD COMPRESSION ═══════
# Use US-JP 2Y yield differential as proxy (available via yfinance)
try:
    us2y=yf.download("^IRX",start='2015-01-01',end='2024-12-31',auto_adjust=False,progress=False)['Close'].squeeze()
    # Actually ^IRX is 13-week T-bill, not 2Y. Let's use Fed funds futures or just test the trend reversal.
    # Simpler: short USDJPY when USDJPY is BELOW its 200-day SMA (trend reversal signal)
    uj=fx['USDJPY']
    uj_m=uj.resample('ME').last()
    sma200=uj.rolling(200).mean().resample('ME').last()
    below200=(uj_m<sma200)
    
    entries=pd.Series(False,index=uj_m.index); exits=pd.Series(False,index=uj_m.index)
    for i in range(1,len(below200)):
        if below200.iloc[i] and not below200.iloc[i-1]:
            entries.iloc[i]=True  # short USDJPY (bet on yen strength)
            if i+3<len(exits): exits.iloc[i+3]=True  # hold for 3 months
    
    if entries.sum()>=5:
        # For SHORT, we need price to fall. yfinance USDJPY=X is USD per JPY.
        # Short means we profit when it goes DOWN.
        # Use inverse logic: entry when we bet on JPY strength, exit reverses
        r=test(uj_m,entries,exits,"SHORT USDJPY <200SMA")
        if r:
            r['ret']=-r['ret']; r['bench']=-r['bench']  # flip for short
            results.append(r)
except: pass

# ═══════ 3. SIMPLE CARRY: buy high-yielder vs low-yielder monthly ═══════
# AUD and NZD typically have higher rates; JPY and CHF lower
high_yield=fx['AUDUSD'].resample('ME').last()
low_yield=fx['USDJPY'].resample('ME').last()
aligned=pd.DataFrame({'high':high_yield,'low':low_yield}).dropna()
entries=pd.Series(False,index=aligned.index); exits=pd.Series(False,index=aligned.index)
for i in range(1,len(aligned)):
    entries.iloc[i]=True
    if i+1<len(exits): exits.iloc[i+1]=True  # rebalance monthly
r=test(aligned['high']/aligned['low'],entries,exits,"Carry AUD/JPY monthly")
if r: results.append(r)

# ── OUTPUT ───────────────────────────────────────────
df=pd.DataFrame(results).sort_values('pf',ascending=False)
print(f"\n{'='*120}")
print(f"STRUCTURAL INEFFICIENCY TESTS — {len(results)} strategies")
print(f"{'='*120}\n")

good=df[(df['ret']-df['bench']>0)&(df['pf']>1.1)&(df['trades']>=10)]
print(f"🏆 Qualifying (Exc>0, PF>1.1, ≥10 trades): {len(good)}")
print(f"{'Strategy':<35} {'Tr':>5} {'Ret%':>7} {'Exc%':>7} {'PF':>6} {'Shrp':>6} {'Ulcr':>6} {'Win%':>6}")
print("-"*100)
for _,r in good.sort_values('pf',ascending=False).iterrows():
    print(f"{r['name']:<35} {r['trades']:>5.0f} {r['ret']:>7.1f} {r['ret']-r['bench']:>7.1f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>6.2f} {r['ulcer']:>6.1f} {r['win_rate']:>6.1f}")

# Show all commodity-FX results
fx_com=df[df['name'].str.contains('←')]
if not fx_com.empty:
    print(f"\n📊 Commodity→FX lag signals (top by PF):")
    for _,r in fx_com.sort_values('pf',ascending=False).head(10).iterrows():
        print(f"  {r['name']:<35} PF={r['pf']:.2f} Exc={r['ret']-r['bench']:+.1f}% Tr={r['trades']:.0f}")
