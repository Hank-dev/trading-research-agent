import sys
import types

import pytest

from trading_research_agent.nodes import generate_portfolio_slate as gps
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec


def make_spec(name: str) -> PortfolioSpec:
    return PortfolioSpec(
        name=name,
        assets=["SPY", "TLT"],
        portfolio_family=PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        start_date="2015-01-01",
        end_date="2023-01-01",
        lookback_days=126,
        top_k=1,
        rebalance_days=21,
        hypothesis="h",
    )


def test_generate_portfolio_slate_rejects_zero_size() -> None:
    with pytest.raises(ValueError):
        gps.generate_portfolio_slate("idea", 0)


def test_portfolio_slate_prompt_includes_volatility_scaled_momentum() -> None:
    assert "volatility_scaled_momentum" in gps.SLATE_PROMPT
    assert "inverse recent volatility" in gps.SLATE_PROMPT


def test_generate_portfolio_slate_returns_specs(monkeypatch) -> None:
    expected = gps.PortfolioSlate(portfolios=[make_spec("A"), make_spec("B")])

    class FakeStructured:
        def invoke(self, _messages):
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(
        gps,
        "load_settings",
        lambda: type("S", (), {"model": "grok-4.3", "api_key": "k", "base_url": "u"})(),
    )
    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = lambda **kwargs: FakeModel()
    sys.modules["langchain_openai"] = fake_module

    result = gps.generate_portfolio_slate("rotate across classes", 2)
    assert [s.name for s in result] == ["A", "B"]


def test_generate_portfolio_slate_truncates_to_requested_size(monkeypatch) -> None:
    expected = gps.PortfolioSlate(
        portfolios=[make_spec("A"), make_spec("B"), make_spec("C")]
    )

    class FakeStructured:
        def invoke(self, _messages):
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(
        gps,
        "load_settings",
        lambda: type("S", (), {"model": "grok-4.3", "api_key": "k", "base_url": "u"})(),
    )
    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = lambda **kwargs: FakeModel()
    sys.modules["langchain_openai"] = fake_module

    result = gps.generate_portfolio_slate("idea", 2)
    assert len(result) == 2
