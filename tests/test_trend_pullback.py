import math

import pandas as pd

from trading_research_agent.strategies.trend_pullback import (
    TrendPullbackParams,
    prepare_indicators,
    simulate_trades,
)


def _synthetic_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=80, freq="D")
    close = [100.0 + i * 0.2 for i in range(80)]
    high = [v + 1.0 for v in close]
    low = [v - 1.0 for v in close]
    open_ = close.copy()

    # Force a Donchian support touch two bars before the entry signal.
    low[60] = 95.0
    close[60] = 108.0
    high[60] = 109.0
    open_[60] = 108.5

    # Signal bar: still above EMA and recovered above current channel.
    close[62] = 114.0
    open_[62] = 113.5
    high[62] = 115.0
    low[62] = 112.0

    # Fill next open, then rally enough to lift the trailing stop.
    open_[63] = 115.0
    close[63] = 116.0
    high[63] = 117.0
    low[63] = 114.5
    close[64] = 125.0
    high[64] = 126.0
    low[64] = 124.0
    open_[64] = 116.0
    open_[65] = 124.0
    high[65] = 125.0
    low[65] = 100.0
    close[65] = 101.0

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        },
        index=index,
    )


def test_prepare_indicators_marks_touch_recovery_signal():
    df = _synthetic_data()
    params = TrendPullbackParams(ema_period=20, sl_atr=1.0, trail_atr=3.0)

    prepared = prepare_indicators(df, params)

    assert bool(prepared.loc["2024-03-03", "raw_entry"])


def test_simulation_fills_next_open_and_charges_round_trip_cost():
    df = _synthetic_data()
    no_cost = TrendPullbackParams(
        ema_period=20,
        sl_atr=1.0,
        trail_atr=3.0,
        cost_bps_per_side=0.0,
    )
    with_cost = TrendPullbackParams(
        ema_period=20,
        sl_atr=1.0,
        trail_atr=3.0,
        cost_bps_per_side=10.0,
    )

    no_cost_trade = simulate_trades(df, no_cost)[0]
    cost_trade = simulate_trades(df, with_cost)[0]

    assert no_cost_trade["entry_signal_date"] == pd.Timestamp("2024-03-03")
    assert no_cost_trade["entry_date"] == pd.Timestamp("2024-03-04")
    assert math.isclose(no_cost_trade["entry_price"], 115.0)
    assert math.isclose(
        no_cost_trade["net_return_pct"] - cost_trade["net_return_pct"],
        0.2,
        abs_tol=1e-12,
    )
