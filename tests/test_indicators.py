import pandas as pd

from trading_research_agent.tools.indicators import (
    donchian_high,
    donchian_low,
    rsi,
    sma,
)


def test_sma_output_length_matches_input_length() -> None:
    values = pd.Series([1, 2, 3, 4, 5])
    assert len(sma(values, 3)) == len(values)


def test_rsi_output_length_matches_input_length() -> None:
    values = pd.Series(range(1, 30))
    assert len(rsi(values, 14)) == len(values)


def test_donchian_high_uses_shifted_previous_channel() -> None:
    high = pd.Series([1, 2, 10, 3])
    result = donchian_high(high, 2)
    assert result.iloc[2] == 2
    assert result.iloc[2] != high.rolling(2).max().iloc[2]


def test_donchian_low_uses_shifted_previous_channel() -> None:
    low = pd.Series([5, 4, 1, 3])
    result = donchian_low(low, 2)
    assert result.iloc[2] == 4
    assert result.iloc[2] != low.rolling(2).min().iloc[2]
