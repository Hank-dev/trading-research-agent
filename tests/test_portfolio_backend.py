import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
pytest.importorskip("vectorbt")

from trading_research_agent.backtesting.backends.portfolio_vectorbt import (
    PortfolioVectorbtBackend,
)
from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec


def make_spec(**overrides) -> PortfolioSpec:
    data = {
        "name": "Rotation",
        "assets": ["AAA", "BBB", "CCC"],
        "portfolio_family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        "start_date": "2020-01-01",
        "end_date": "2022-06-01",
        "lookback_days": 20,
        "top_k": 1,
        "rebalance_days": 21,
        "hypothesis": "Relative strength persists.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


def make_panel(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    aaa = pd.Series(np.linspace(100, 500, rows), index=index)
    bbb = pd.Series(100 + 10 * np.sin(np.linspace(0, 20, rows)), index=index)
    ccc = pd.Series(np.linspace(100, 60, rows), index=index)
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc})


def test_portfolio_backend_runs_and_returns_result() -> None:
    result = PortfolioVectorbtBackend().run(make_spec(), make_panel())

    assert isinstance(result, BacktestResult)
    assert result.engine == "vectorbt_portfolio"
    assert result.asset.startswith("PORTFOLIO[")
    assert result.metrics.final_equity > 0
    assert any(
        check.test_name == "Portfolio rebalance count"
        for check in result.robustness_results
    )


def test_portfolio_backend_beats_equal_weight_on_clear_trend() -> None:
    # AAA strongly outperforms; a momentum rotation that concentrates in AAA
    # should beat the equal-weight benchmark that's dragged down by CCC.
    result = PortfolioVectorbtBackend().run(make_spec(top_k=1), make_panel())
    assert result.metrics.total_return_pct > result.metrics.buy_and_hold_return_pct
    assert result.metrics.beats_benchmark is True
