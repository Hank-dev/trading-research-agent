from math import sqrt

import pytest

from trading_research_agent.tools.stats import (
    deflated_sharpe_ratio,
    deflated_sharpe_threshold,
    estimate_trading_days,
    probabilistic_sharpe_ratio,
)


def test_estimate_trading_days_roughly_one_year() -> None:
    days = estimate_trading_days("2020-01-01", "2021-01-01")
    # ~365 calendar days * 252/365 = 252
    assert 248 <= days <= 256


def test_estimate_trading_days_minimum_two() -> None:
    assert estimate_trading_days("2020-01-01", "2020-01-01") == 2


def test_psr_zero_sharpe_is_50_percent() -> None:
    psr = probabilistic_sharpe_ratio(0.0, n_obs=252)
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_high_sharpe_long_period_approaches_one() -> None:
    psr = probabilistic_sharpe_ratio(2.0, n_obs=252 * 5)
    assert psr > 0.99


def test_psr_negative_sharpe_below_half() -> None:
    psr = probabilistic_sharpe_ratio(-1.0, n_obs=252)
    assert psr < 0.5


def test_psr_more_data_makes_same_sharpe_more_significant() -> None:
    short = probabilistic_sharpe_ratio(1.0, n_obs=63)
    long = probabilistic_sharpe_ratio(1.0, n_obs=252 * 4)
    assert long > short


def test_psr_handles_too_few_observations() -> None:
    assert probabilistic_sharpe_ratio(1.0, n_obs=1) == 0.0


def test_deflated_sharpe_threshold_grows_with_trials() -> None:
    variance_daily = (0.5 / sqrt(252)) ** 2
    threshold_5 = deflated_sharpe_threshold(variance_daily, n_trials=5)
    threshold_50 = deflated_sharpe_threshold(variance_daily, n_trials=50)
    assert threshold_50 > threshold_5 > 0


def test_deflated_sharpe_threshold_zero_with_fewer_than_two_trials() -> None:
    assert deflated_sharpe_threshold(1.0, n_trials=1) == 0.0


def test_dsr_falls_back_to_psr_with_single_trial() -> None:
    psr = probabilistic_sharpe_ratio(1.0, n_obs=504)
    dsr = deflated_sharpe_ratio(1.0, n_obs=504, annual_sharpes_across_trials=[1.0])
    assert dsr == pytest.approx(psr)


def test_dsr_lower_than_psr_when_many_trials_have_high_variance() -> None:
    psr = probabilistic_sharpe_ratio(1.0, n_obs=504)
    dsr = deflated_sharpe_ratio(
        1.0,
        n_obs=504,
        annual_sharpes_across_trials=[1.0, 0.8, 0.6, 0.4, -0.2, -0.5, 1.2, 0.9],
    )
    assert dsr < psr


def test_dsr_close_to_psr_when_slate_sharpes_are_identical() -> None:
    psr = probabilistic_sharpe_ratio(1.0, n_obs=504)
    dsr = deflated_sharpe_ratio(
        1.0, n_obs=504, annual_sharpes_across_trials=[1.0, 1.0, 1.0, 1.0]
    )
    # Zero variance across trials -> threshold = 0 -> DSR == PSR
    assert dsr == pytest.approx(psr)
