from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.workflows import researched_hypothesis_generator as rhg


def _spec(name: str, family: PortfolioFamily = PortfolioFamily.DUAL_MOMENTUM) -> PortfolioSpec:
    return PortfolioSpec(
        name=name,
        assets=["SPY", "TLT", "GLD"],
        portfolio_family=family,
        start_date="2010-01-01",
        end_date="2024-01-01",
        lookback_days=126,
        top_k=1,
        rebalance_days=21,
        hypothesis="Fallback hypothesis.",
    )


def test_fallback_chain_calls_fireworks_before_codex(monkeypatch) -> None:
    """Fireworks fallback fires after primary LLM fails, but before Codex."""
    kwargs_passed = {}
    fireworks_called = False
    codex_called = False

    def fake_fireworks(**kwargs):
        nonlocal fireworks_called
        fireworks_called = True
        kwargs_passed.update(kwargs)
        return rhg.ResearchedPortfolioSlate(
            research_brief="Fireworks fallback brief.",
            hypotheses=[
                rhg.ResearchHypothesis(
                    title="Fw hypothesis",
                    mechanism="Fireworks DeepSeek V4 Pro explains anomalies.",
                    evidence_to_check=["anomaly persists"],
                    falsification_tests=["lockbox fail"],
                    portfolio_index=0,
                )
            ],
            portfolios=[_spec("fw fallback")],
        )

    def fake_codex(**kwargs):
        nonlocal codex_called
        codex_called = True
        raise RuntimeError("should not reach Codex")

    monkeypatch.setattr(rhg, "_invoke_research_model", lambda **kw: (_ for _ in ()).throw(RuntimeError("primary failed")))
    monkeypatch.setattr(rhg, "_invoke_fireworks_model", fake_fireworks)
    monkeypatch.setattr(rhg, "_invoke_codex_model", fake_codex)

    result = rhg.generate_researched_portfolio_slate(
        user_request="macro",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
        anomaly_facts=["DBC leads TLT at lag 5."],
    )

    assert fireworks_called
    assert not codex_called
    assert result.research_brief == "Fireworks fallback brief."
    assert result.portfolios[0].name == "fw fallback"


def test_fallback_chain_reaches_codex_when_both_primary_and_fireworks_fail(monkeypatch) -> None:
    fireworks_called = False
    codex_called = False

    def fake_fireworks(**kwargs):
        nonlocal fireworks_called
        fireworks_called = True
        raise RuntimeError("Fireworks also failed")

    def fake_codex(**kwargs):
        nonlocal codex_called
        codex_called = True
        return rhg.ResearchedPortfolioSlate(
            research_brief="Codex fallback.",
            hypotheses=[
                rhg.ResearchHypothesis(
                    title="Codex final",
                    mechanism="Codex fallback saves the day.",
                    evidence_to_check=["all"],
                    falsification_tests=["all"],
                    portfolio_index=0,
                )
            ],
            portfolios=[_spec("codex final")],
        )

    monkeypatch.setattr(rhg, "_invoke_research_model", lambda **kw: (_ for _ in ()).throw(RuntimeError("primary failed")))
    monkeypatch.setattr(rhg, "_invoke_fireworks_model", fake_fireworks)
    monkeypatch.setattr(rhg, "_invoke_codex_model", fake_codex)

    result = rhg.generate_researched_portfolio_slate(
        user_request="macro",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
    )

    assert fireworks_called
    assert codex_called
    assert result.portfolios[0].name == "codex final"


def test_fireworks_model_rejects_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FIREWORKS_FALLBACK_API_KEY", raising=False)
    monkeypatch.setattr(rhg, "_invoke_research_model", lambda **kw: (_ for _ in ()).throw(RuntimeError("primary failed")))
    monkeypatch.setattr(rhg, "shutil", type("S", (), {"which": lambda x: None})())

    try:
        rhg.generate_researched_portfolio_slate(
            user_request="macro",
            assets=["SPY", "TLT", "GLD"],
            start="2010-01-01",
            end="2024-01-01",
            slate_size=1,
        )
    except RuntimeError as exc:
        err = str(exc).lower()
        assert "fireworks" in err or "primary" in err or "codex" in err
    else:
        raise AssertionError("expected RuntimeError")
