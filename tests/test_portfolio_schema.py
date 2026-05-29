import pytest
from pydantic import ValidationError

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec


def make(**overrides) -> dict:
    data = {
        "name": "P",
        "assets": ["SPY", "VIXY"],
        "portfolio_family": PortfolioFamily.CRISIS_HEDGE,
        "start_date": "2015-01-01",
        "end_date": "2024-01-01",
        "lookback_days": 100,
        "hedge_weight": 0.2,
        "hypothesis": "Tail hedge.",
    }
    data.update(overrides)
    return data


def test_crisis_hedge_valid() -> None:
    spec = PortfolioSpec(**make())
    assert spec.hedge_weight == 0.2
    assert len(spec.assets) == 2


def test_crisis_hedge_requires_hedge_weight() -> None:
    with pytest.raises(ValidationError, match="hedge_weight"):
        PortfolioSpec(**make(hedge_weight=None))


def test_crisis_hedge_caps_hedge_weight() -> None:
    with pytest.raises(ValidationError, match="hedge_weight"):
        PortfolioSpec(**make(hedge_weight=0.8))


def test_crisis_hedge_requires_exactly_two_assets() -> None:
    with pytest.raises(ValidationError, match="exactly 2 assets"):
        PortfolioSpec(**make(assets=["SPY", "TLT", "VIXY"]))


def test_time_series_momentum_does_not_require_hedge_weight() -> None:
    spec = PortfolioSpec(
        **make(
            portfolio_family=PortfolioFamily.TIME_SERIES_MOMENTUM,
            assets=["SPY", "TLT", "GLD"],
            hedge_weight=None,
        )
    )
    assert spec.portfolio_family == PortfolioFamily.TIME_SERIES_MOMENTUM
