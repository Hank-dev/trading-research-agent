#!/usr/bin/env python3
"""
Track daily J225 (Nikkei 225) Trend-Pullback Paper Trading.
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# Insert src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_research_agent.strategies.trend_pullback import (
    TrendPullbackParams,
    prepare_indicators,
    simulate_trades,
    summarize_trades
)

def main():
    ticker = "^N225"
    print(f"Loading data for J225 ({ticker})...")
    
    # Download data from 2010-01-01 to today
    data = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)
    if data.empty:
        print(f"Error: No data downloaded for {ticker}")
        sys.exit(1)
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    print(f"Data range: {data.index[0].date()} to {data.index[-1].date()} ({len(data)} trading days)")

    # Define best parameter family from J225 research:
    params = TrendPullbackParams(
        side="long",
        ema_period=100,
        donchian_window=21,
        delay_bars=2,
        atr_window=20,
        sl_atr=1.0,
        trail_atr=3.0,
        cost_bps_per_side=10.0
    )

    # 1. Run simulation
    trades = simulate_trades(data, params)
    
    # Prepare indicators so we can extract live stops
    prepared = prepare_indicators(data, params)
    
    summary = summarize_trades(trades)

    print("\n" + "="*60)
    print("📈 J225 DAILY LONG TREND-PULLBACK PERFORMANCE (2010-2026)")
    print("="*60)
    print(f"Total Trades:      {summary['trades']}")
    print(f"Total Net Return:  {summary['net_return_pct']:.1f}%")
    print(f"Total R-Multiple:  {summary['total_r']:.1f} R")
    print(f"Average R:         {summary['avg_r']:.2f} R")
    print(f"Win Rate:          {summary['win_rate']:.1f}%")
    print(f"Profit Factor:     {summary['profit_factor']:.2f}")
    print(f"Max Drawdown:      {summary['max_dd_pct']:.1f}%")
    print(f"Avg Hold Bars:     {summary['avg_hold_bars']:.1f} days")
    print("="*60)

    # 2. Extract current live state
    if not trades:
        print("Status: No trades simulated yet.")
        sys.exit(0)

    last_trade = trades[-1]
    
    # Check if the last trade is still open
    is_open = last_trade["exit_reason"] == "end_of_data"
    
    print("\n" + "="*60)
    print("📊 CURRENT PAPER POSITION STATUS")
    print("="*60)
    
    if is_open:
        print("🟢 STATUS: ACTIVE LONG POSITION")
        print(f"Entry Signal Date: {last_trade['entry_signal_date'].date()}")
        print(f"Position Entry:    {last_trade['entry_date'].date()} @ {last_trade['entry_price']:.2f}")
        print(f"Current Close:     {last_trade['exit_price']:.2f}")
        print(f"Open Return:       {last_trade['net_return_pct']:.2f}% (net of 10bps entry cost)")
        print(f"Open R-Multiple:   {last_trade['r_multiple']:.2f} R")
        print(f"Bars Held:         {last_trade['bars_held']} days")
        
        # Calculate live stops
        # We find indicators for the signal bar
        entry_idx = last_trade['entry_date']
        signal_idx = last_trade['entry_signal_date']
        
        # Get values at entry signal bar
        entry_atr = float(prepared.loc[signal_idx, "atr"])
        entry_price = last_trade['entry_price']
        
        hard_stop = entry_price - params.sl_atr * entry_atr
        
        # Recalculate trailing stop based on close values since entry
        entry_date_ts = pd.Timestamp(last_trade['entry_date'])
        closed_prices_since_entry = prepared.loc[entry_date_ts:, "Close"].to_numpy()
        
        anchor_close = entry_price
        trail_stop = entry_price - params.trail_atr * entry_atr
        for cp in closed_prices_since_entry[:-1]:  # up to prior bar
            if cp > anchor_close:
                anchor_close = cp
                trail_stop = max(trail_stop, anchor_close - params.trail_atr * entry_atr)
                
        # Current bar low touch check is done inside the loop, we report stops for the current bar:
        print(f"Hard Stop-Loss:    {hard_stop:.2f}")
        print(f"Trailing Stop:     {trail_stop:.2f}")
        print(f"Active Stop-Level: {max(hard_stop, trail_stop):.2f}")
        
    else:
        print("⚪ STATUS: IN CASH (NO ACTIVE POSITION)")
        print(f"Last Position Entry: {last_trade['entry_date'].date()} @ {last_trade['entry_price']:.2f}")
        print(f"Last Position Exit:  {last_trade['exit_date'].date()} @ {last_trade['exit_price']:.2f}")
        print(f"Last Position Net:   {last_trade['net_return_pct']:.2f}%")
        print(f"Last Exit Reason:    {last_trade['exit_reason']}")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
