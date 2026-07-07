"""Stress-test SMA 5/15 across FX, commodities, multiple periods with lockbox."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np; np.random.seed(42)
import pandas as pd; import vectorbt as vbt; import yfinance as yf
from scipy.stats import skew as scipy_skew; import warnings

def ulcer(eq):
    pk=eq.cummax(); dd=100*(eq-pk)/pk; return float(np.sqrt((dd**2).mean()))

def test_strat(close, name, period, fast=5, slow=15):
    sf=close.rolling(fast).mean(); ss=close.rolling(slow).mean()
    entries=(sf>ss)&(sf.shift(1)<=ss.shift(1))
    exits=(sf<ss)&(sf.shift(1)>=ss.shift(1))
    
    n=len(close); split=int(n*0.8)
    ci,ce,cx=close.iloc[:split],entries.iloc[:split],exits.iloc[:split]
    cl,el,xl=close.iloc[split:],entries.iloc[split:],exits.iloc[split:]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pis=vbt.Portfolio.from_signals(ci,ce,cx,freq='D'); si=pis.stats()
        plb=vbt.Portfolio.from_signals(cl,el,xl,freq='D'); sl=plb.stats()
    
    t=pis.trades; w=t.winning.returns.sum() if t.winning.count()>0 else 0
    l=abs(t.losing.returns.sum()) if t.losing.count()>0 else 0.0001
    ir=float(si['Total Return [%]']); ib=float(si['Benchmark Return [%]'])
    lr=float(sl['Total Return [%]']); lb=float(sl['Benchmark Return [%]'])
    ui=ulcer(pis.value())
    
    # Walk-forward
    nwf=5; fsize=len(ci)//nwf; wf=[]
    for i in range(nwf):
        end=fsize*(i+1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pfw=vbt.Portfolio.from_signals(ci.iloc[:end],ce.iloc[:end],cx.iloc[:end],freq='D')
            ss2=pfw.stats()
        ret=float(ss2['Total Return [%]']); bench=float(ss2['Benchmark Return [%]'])
        wf.append(ret-bench)
    wa=np.array(wf)
    
    return {
        'name':name,'period':period,'trades':len(t),'lb_trades':len(plb.trades),
        'is_ret':ir,'is_bench':ib,'is_pf':w/l,'is_sharpe':float(si['Sharpe Ratio']),
        'is_ulcer':ui,'is_mdd':float(si['Max Drawdown [%]']),
        'lb_ret':lr,'lb_bench':lb,'lb_exc':lr-lb,'lb_sharpe':float(sl['Sharpe Ratio']),
        'wf_mean':wa.mean(),'wf_min':wa.min(),'wf_all_pos':int(wa.min()>0)
    }

# ── Assets ──────────────────────────────────────────
ASSETS={
    'FX':{'USDJPY':'USDJPY=X','EURUSD':'EURUSD=X','GBPUSD':'GBPUSD=X',
          'AUDUSD':'AUDUSD=X','USDCHF':'USDCHF=X','USDCAD':'USDCAD=X','NZDUSD':'NZDUSD=X'},
    'Commodities':{'Gold':'GC=F','Oil':'CL=F','Silver':'SI=F','Copper':'HG=F',
                   'NatGas':'NG=F','Corn':'ZC=F','Wheat':'ZW=F'},
    'Indices':{'SP500':'^GSPC','Nasdaq':'^IXIC','DAX':'^GDAXI','FTSE':'^FTSE'},
    'Crypto':{'BTC':'BTC-USD','ETH':'ETH-USD'},
}
PERIODS=[('2015-01-01','2024-12-31','2015-24'),
         ('2020-01-01','2024-12-31','2020-24'),
         ('2010-01-01','2019-12-31','2010-19')]

results=[]

for cat, assets in ASSETS.items():
    for name, sym in assets.items():
        for start, end, plabel in PERIODS:
            try:
                c=yf.download(sym,start=start,end=end,auto_adjust=False,progress=False)['Close'].squeeze()
                if c.empty or len(c)<500: continue
                r=test_strat(c,f"{name}",plabel)
                r['asset']=name; r['cat']=cat; results.append(r)
            except: continue

df=pd.DataFrame(results)

# ── Output ──────────────────────────────────────────
print(f"\n{'='*130}")
print(f"SMA 5/15 STRESS TEST — {len(df)} asset×period combos (lockbox=last 20%)")
print(f"{'='*130}\n")

# Lockbox winners
lbwin=df[df['lb_exc']>0].sort_values('lb_exc',ascending=False)
print(f"✅ Lockbox PASS (positive excess, unseen data): {len(lbwin)}/{len(df)}")
print(f"{'Asset':<12} {'Period':<10} {'Cat':<14} {'IS Tr':>5} {'IS Ret%':>8} {'IS PF':>6} {'LB Tr':>5} {'LB Exc%':>7} {'LB Shrp':>6} {'WF All+':>6} {'IS Ulcr':>6}")
print("-"*120)
for _,r in lbwin.iterrows():
    print(f"{r['asset']:<12} {r['period']:<10} {r['cat']:<14} {r['trades']:>5.0f} {r['is_ret']:>8.0f} {r['is_pf']:>6.2f} "
          f"{r['lb_trades']:>5.0f} {r['lb_exc']:>7.0f} {r['lb_sharpe']:>6.2f} "
          f"{'✅' if r['wf_all_pos'] else '❌':>6} {r['is_ulcer']:>6.1f}")

# Failures with high in-sample PF
bad=df[(df['lb_exc']<0)&(df['is_pf']>1.3)]
print(f"\n⚠️  Trap strategies (IS PF>1.3 but FAIL lockbox): {len(bad)}")
for _,r in bad.sort_values('is_pf',ascending=False).iterrows():
    print(f"  {r['asset']:<12} {r['period']:<10} {r['cat']:<14} IS PF={r['is_pf']:.2f} LB Exc={r['lb_exc']:.0f}%")

# Summary by category
print(f"\n📊 Lockbox pass rate by category:")
for cat in ['FX','Commodities','Indices','Crypto']:
    catdf=df[df['cat']==cat]
    passes=sum(catdf['lb_exc']>0)
    print(f"  {cat:<14}: {passes}/{len(catdf)} pass ({passes/max(len(catdf),1)*100:.0f}%)")

# By period
print(f"\n📊 Lockbox pass rate by period:")
for p in ['2015-24','2020-24','2010-19']:
    pdf=df[df['period']==p]
    passes=sum(pdf['lb_exc']>0)
    print(f"  {p:<10}: {passes}/{len(pdf)} pass ({passes/max(len(pdf),1)*100:.0f}%)")

# All-weather (passes lockbox + walk-forward)
god=df[(df['lb_exc']>0)&(df['wf_all_pos']==1)]
print(f"\n🏆 ALL-WEATHER (lockbox ✅ + walk-forward all-positive ✅): {len(god)}")
for _,r in god.sort_values('lb_exc',ascending=False).iterrows():
    print(f"  {r['asset']:<12} {r['period']:<10} {r['cat']:<14} IS PF={r['is_pf']:.2f} LB Exc={r['lb_exc']:+.0f}% IS Ulcer={r['is_ulcer']:.1f}")
