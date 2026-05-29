import numpy as np
import pandas as pd


def _to_series(values: pd.Series) -> pd.Series:
    return pd.Series(values, dtype="float64")


def sma(series: pd.Series, window: int) -> pd.Series:
    return _to_series(series).rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    prices = _to_series(series)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - (100 / (1 + rs))
    values = values.where(avg_loss != 0, 100.0)
    values = values.where(avg_gain != 0, 0.0)
    return values


def donchian_high(high: pd.Series, window: int) -> pd.Series:
    return _to_series(high).rolling(window).max().shift(1)


def donchian_low(low: pd.Series, window: int) -> pd.Series:
    return _to_series(low).rolling(window).min().shift(1)
