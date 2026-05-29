"""Probabilistic and deflated Sharpe ratio helpers.

Both follow the López de Prado / Bailey formulation under the normality
assumption: we do not have access to the raw returns series here, so skew
and kurtosis terms are omitted.

References:
- Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier"
- López de Prado (2018), "Advances in Financial Machine Learning", ch. 11
"""

from datetime import date
from math import e, sqrt
from statistics import NormalDist


_EULER_MASCHERONI = 0.5772156649015329
_TRADING_DAYS_PER_YEAR = 252
_NORMAL = NormalDist()


def estimate_trading_days(start_iso: str, end_iso: str) -> int:
    """Approximate trading days in a date range. ~252/365 calendar days are trading days."""
    days = (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days
    if days < 2:
        return 2
    return max(2, round(days * _TRADING_DAYS_PER_YEAR / 365))


def probabilistic_sharpe_ratio(
    annual_sharpe: float,
    n_obs: int,
    sr_threshold_daily: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds `sr_threshold_daily`, given the
    observed annualized Sharpe and number of daily observations.

    Returns a value in [0, 1]. Assumes returns are i.i.d. normal — fine for an
    MVP-quality robustness signal, not for a regulator.
    """
    if n_obs < 2:
        return 0.0
    daily_sr = annual_sharpe / sqrt(_TRADING_DAYS_PER_YEAR)
    return _NORMAL.cdf((daily_sr - sr_threshold_daily) * sqrt(n_obs - 1))


def deflated_sharpe_threshold(sharpe_variance_daily: float, n_trials: int) -> float:
    """Expected maximum daily Sharpe across `n_trials` independent trials, under
    the López de Prado approximation. Used as the null-hypothesis Sharpe in DSR."""
    if n_trials < 2 or sharpe_variance_daily <= 0:
        return 0.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * e))
    return sqrt(sharpe_variance_daily) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe_ratio(
    annual_sharpe: float,
    n_obs: int,
    annual_sharpes_across_trials: list[float],
) -> float:
    """Probability that this candidate's Sharpe survives correction for the
    number of trials in the slate. Returns a value in [0, 1].

    Requires at least two trials and non-zero variance across them; otherwise
    returns PSR with threshold 0.
    """
    n_trials = len(annual_sharpes_across_trials)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(annual_sharpe, n_obs)

    daily_sharpes = [s / sqrt(_TRADING_DAYS_PER_YEAR) for s in annual_sharpes_across_trials]
    mean = sum(daily_sharpes) / n_trials
    variance = sum((s - mean) ** 2 for s in daily_sharpes) / (n_trials - 1)
    threshold = deflated_sharpe_threshold(variance, n_trials)
    return probabilistic_sharpe_ratio(annual_sharpe, n_obs, sr_threshold_daily=threshold)
