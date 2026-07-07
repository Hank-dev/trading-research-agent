import json

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
        hypothesis="A falsifiable structural hypothesis.",
    )


def test_researched_slate_contains_research_hypotheses_and_portfolios(monkeypatch) -> None:
    fake = rhg.ResearchedPortfolioSlate(
        research_brief="Rates, inflation, crisis beta, and commodity shocks imply distinct rotation mechanisms.",
        hypotheses=[
            rhg.ResearchHypothesis(
                title="Inflation shock rotation",
                mechanism="Commodity-sensitive assets may lead inflation hedges during positive price shocks.",
                evidence_to_check=["commodity trend", "bond drawdown", "gold relative strength"],
                falsification_tests=["lockbox fail", "leave-one-asset-out fail"],
                portfolio_index=0,
            )
        ],
        portfolios=[_spec("researched dual momentum")],
    )
    monkeypatch.setattr(rhg, "_invoke_research_model", lambda *args, **kwargs: fake)

    result = rhg.generate_researched_portfolio_slate(
        user_request="Find structurally motivated macro rotation strategies.",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
    )

    assert "Rates" in result.research_brief
    assert result.hypotheses[0].portfolio_index == 0
    assert result.portfolios[0].start_date == "2010-01-01"
    assert result.portfolios[0].end_date == "2024-01-01"
    assert result.portfolios[0].assets == ["SPY", "TLT", "GLD"]


def test_researched_slate_rejects_missing_hypothesis_mapping(monkeypatch) -> None:
    fake = {
        "research_brief": "brief",
        "hypotheses": [],
        "portfolios": [_spec("orphan").model_dump()],
    }
    monkeypatch.setattr(rhg, "_invoke_research_model", lambda *args, **kwargs: fake)

    try:
        rhg.generate_researched_portfolio_slate(
            user_request="macro",
            assets=["SPY", "TLT", "GLD"],
            start="2010-01-01",
            end="2024-01-01",
            slate_size=1,
        )
    except ValueError as exc:
        assert "hypothes" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_researched_slate_passes_anomaly_facts_to_research_model(monkeypatch) -> None:
    captured = {}
    fake = rhg.ResearchedPortfolioSlate(
        research_brief="brief",
        hypotheses=[
            rhg.ResearchHypothesis(
                title="Gold leads NZD",
                mechanism="Liquidity shock transmission.",
                evidence_to_check=["lag stability"],
                falsification_tests=["lockbox fail"],
                portfolio_index=0,
            )
        ],
        portfolios=[_spec("lagged gold filter")],
    )

    def fake_invoke(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(rhg, "_invoke_research_model", fake_invoke)

    rhg.generate_researched_portfolio_slate(
        user_request="Find non-generic macro hypotheses.",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
        anomaly_facts=["GLD 20d returns lead NZDUSD at lag 20 with corr +0.31."],
    )

    assert captured["anomaly_facts"] == ["GLD 20d returns lead NZDUSD at lag 20 with corr +0.31."]


def test_researched_slate_falls_back_to_codex_when_primary_llm_fails(monkeypatch) -> None:
    fake = rhg.ResearchedPortfolioSlate(
        research_brief="Codex fallback brief.",
        hypotheses=[
            rhg.ResearchHypothesis(
                title="Codex hypothesis",
                mechanism="Fallback explains concrete anomaly facts.",
                evidence_to_check=["fact persists"],
                falsification_tests=["lockbox fail"],
                portfolio_index=0,
            )
        ],
        portfolios=[_spec("codex fallback spec")],
    )
    monkeypatch.setattr(rhg, "_invoke_research_model", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("401 auth")))
    monkeypatch.setattr(rhg, "_invoke_codex_model", lambda **kwargs: fake)

    result = rhg.generate_researched_portfolio_slate(
        user_request="macro",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
        anomaly_facts=["DBC leads TLT at lag 5."],
    )

    assert result.research_brief == "Codex fallback brief."
    assert result.portfolios[0].name == "codex fallback spec"


def test_codex_json_parser_accepts_fenced_json() -> None:
    payload = {
        "research_brief": "brief",
        "hypotheses": [
            {
                "title": "h",
                "mechanism": "m",
                "evidence_to_check": ["e"],
                "falsification_tests": ["f"],
                "portfolio_index": 0,
            }
        ],
        "portfolios": [_spec("parsed").model_dump(mode="json")],
    }
    text = "```json\n" + json.dumps(payload) + "\n```"

    parsed = rhg._parse_codex_json(text)

    assert parsed.research_brief == "brief"
    assert parsed.portfolios[0].name == "parsed"


def test_codex_subprocess_env_strips_app_llm_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "bad-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("XAI_API_KEY", "bad-xai")

    env = rhg._codex_subprocess_env()

    assert "OPENAI_API_KEY" not in env
    assert "LLM_PROVIDER" not in env
    assert "XAI_API_KEY" not in env
    assert env["NO_COLOR"] == "1"


def test_codex_error_summary_identifies_missing_auth() -> None:
    text = rhg._summarize_codex_error("unexpected status 401 Unauthorized: Missing bearer")

    assert "codex login" in text
    assert "not authenticated" in text


def test_researched_slate_normalizes_dates_and_asset_universe(monkeypatch) -> None:
    fake = rhg.ResearchedPortfolioSlate(
        research_brief="brief",
        hypotheses=[
            rhg.ResearchHypothesis(
                title="x",
                mechanism="y",
                evidence_to_check=["z"],
                falsification_tests=["lockbox"],
                portfolio_index=0,
            )
        ],
        portfolios=[
            _spec("messy", PortfolioFamily.CROSS_SECTIONAL_MOMENTUM).model_copy(
                update={"assets": ["SPY", "BTC-USD"], "start_date": "2000-01-01", "end_date": "2030-01-01"}
            )
        ],
    )
    monkeypatch.setattr(rhg, "_invoke_research_model", lambda *args, **kwargs: fake)

    result = rhg.generate_researched_portfolio_slate(
        user_request="macro",
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        slate_size=1,
    )

    spec = result.portfolios[0]
    assert spec.assets == ["SPY", "TLT", "GLD"]
    assert spec.start_date == "2010-01-01"
    assert spec.end_date == "2024-01-01"
