from trading_research_agent.nodes import suggest_history_directions
from trading_research_agent.nodes.suggest_history_directions import (
    HistorySuggestion,
    format_history_prompt,
    suggest_directions_from_history,
)


def make_summary(**overrides) -> dict:
    base = {
        "total_trials": 12,
        "lockbox_runs": 2,
        "by_verdict": {"reject": 8, "needs_more_testing": 4},
        "by_asset": {"BTC-USD": 6, "QQQ": 4, "SPY": 2},
        "by_family": {"sma_crossover": 7, "donchian_breakout": 5},
        "asset_family_pairs": {
            "BTC-USD / sma_crossover": 4,
            "QQQ / sma_crossover": 3,
            "BTC-USD / donchian_breakout": 2,
        },
        "failed_checks": {
            "Benchmark comparison": 9,
            "Deflated Sharpe ratio (DSR)": 8,
            "Sharpe ratio significance (PSR)": 4,
        },
        "passed_runs": [],
    }
    base.update(overrides)
    return base


def test_format_prompt_includes_key_aggregates() -> None:
    prompt = format_history_prompt(make_summary())
    assert "Total trials across all sessions: 12" in prompt
    assert "Held-out lockbox verifications: 2" in prompt
    assert "9x Benchmark comparison" in prompt
    assert "BTC-USD" in prompt
    assert "sma_crossover" in prompt


def test_format_prompt_lists_passing_trials_when_any() -> None:
    summary = make_summary(
        passed_runs=[
            {
                "asset": "QQQ",
                "strategy_family": "sma_crossover",
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
            }
        ]
    )
    prompt = format_history_prompt(summary)
    assert "Passing trials:" in prompt
    assert "QQQ sma_crossover" in prompt


def test_format_prompt_handles_empty_counters() -> None:
    summary = make_summary(
        total_trials=0,
        by_verdict={},
        by_asset={},
        by_family={},
        asset_family_pairs={},
        failed_checks={},
    )
    prompt = format_history_prompt(summary)
    assert "(none recorded)" in prompt


def test_suggest_directions_returns_parsed_model(monkeypatch) -> None:
    expected = HistorySuggestion(
        summary="You have tried many simple rules; none have passed lockbox.",
        structural_gaps=["Multi-asset rotation", "Volatility regime filters"],
        next_directions=[
            "Try a different asset class such as commodities or rates.",
            "Stop testing parameter variations of trend-following.",
        ],
        honest_warnings=[
            "Cumulative trial count is high; the DSR bar is now very strict.",
            "Consider stopping the search and building richer signals.",
        ],
    )

    class FakeStructured:
        def invoke(self, _messages):
            return expected

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructured()

    monkeypatch.setattr(
        suggest_history_directions,
        "load_settings",
        lambda: type("S", (), {"model": "grok-4.3", "api_key": "k", "base_url": "u"})(),
    )
    # Patch the LangChain import to return our fake.
    import sys
    import types

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = lambda **kwargs: FakeModel()
    sys.modules["langchain_openai"] = fake_module

    result = suggest_directions_from_history(make_summary())
    assert result.summary == expected.summary
    assert "Multi-asset rotation" in result.structural_gaps


def test_history_suggestion_caps_next_directions_at_three() -> None:
    # Schema enforces max_length=3
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HistorySuggestion(
            summary="x",
            structural_gaps=[],
            next_directions=["a", "b", "c", "d"],
            honest_warnings=[],
        )
