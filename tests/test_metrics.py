import pandas as pd

from trading_research_agent.tools.metrics import (
    calculate_buy_and_hold_return,
    extract_backtesting_py_metrics,
)


def test_buy_and_hold_return_is_calculated_correctly() -> None:
    data = pd.DataFrame({"Close": [100.0, 110.0]})
    assert calculate_buy_and_hold_return(data) == 10.000000000000009


def test_beats_benchmark_only_when_strategy_exceeds_benchmark() -> None:
    stats = pd.Series(
        {
            "Return [%]": 11.0,
            "Sharpe Ratio": 1.2,
            "Max. Drawdown [%]": -12.5,
            "# Trades": 5,
            "Win Rate [%]": 60.0,
            "Exposure Time [%]": 40.0,
            "Equity Final [$]": 11100.0,
        }
    )

    metrics = extract_backtesting_py_metrics(stats, buy_and_hold_return_pct=10.0)
    assert metrics.beats_benchmark is True

    metrics = extract_backtesting_py_metrics(stats, buy_and_hold_return_pct=12.0)
    assert metrics.beats_benchmark is False


def test_missing_sharpe_ratio_does_not_crash() -> None:
    stats = pd.Series(
        {
            "Return [%]": 11.0,
            "Sharpe Ratio": float("nan"),
            "Max. Drawdown [%]": -12.5,
            "# Trades": 5,
            "Win Rate [%]": 60.0,
            "Exposure Time [%]": 40.0,
            "Equity Final [$]": 11100.0,
        }
    )

    metrics = extract_backtesting_py_metrics(stats, buy_and_hold_return_pct=10.0)

    assert metrics.sharpe_ratio is None


def test_drawdown_sign_is_preserved() -> None:
    stats = pd.Series(
        {
            "Return [%]": 11.0,
            "Max. Drawdown [%]": -12.5,
            "# Trades": 5,
            "Equity Final [$]": 11100.0,
        }
    )

    metrics = extract_backtesting_py_metrics(stats, buy_and_hold_return_pct=10.0)

    assert metrics.max_drawdown_pct == -12.5
