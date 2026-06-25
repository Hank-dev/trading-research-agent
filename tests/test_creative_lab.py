from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.workflows import creative_lab as cl
from trading_research_agent.workflows.anomaly_miner import AnomalyFact


def _state(spec: PortfolioSpec, verdict: str, ret: float = 20.0, sharpe: float = 0.8) -> dict:
    return {
        "strategy_spec": spec,
        "backtest_result": BacktestResult(
            strategy_name=spec.name,
            asset="PORTFOLIO[...]",
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine="vectorbt_portfolio",
            metrics=BacktestMetrics(
                total_return_pct=ret,
                buy_and_hold_return_pct=10.0,
                sharpe_ratio=sharpe,
                max_drawdown_pct=-12.0,
                num_trades=30,
                win_rate_pct=55.0,
                exposure_time_pct=70.0,
                final_equity=12000.0,
                beats_benchmark=ret > 10.0,
            ),
        ),
        "report": ResearchReport(markdown="x", verdict=verdict, reasons=[], next_tests=[]),
    }


def test_generate_creative_slate_is_preregistered_diverse_and_bounded() -> None:
    slate = cl.generate_creative_slate(
        assets=["SPY", "TLT", "GLD", "DBC"],
        start="2010-01-01",
        end="2024-01-01",
        max_candidates=7,
    )

    assert len(slate) == 7
    assert len({spec.name for spec in slate}) == 7
    assert len({spec.portfolio_family for spec in slate}) >= 4
    assert all(spec.assets == ["SPY", "TLT", "GLD", "DBC"] for spec in slate)
    assert all(spec.start_date == "2010-01-01" for spec in slate)
    assert all(spec.end_date == "2024-01-01" for spec in slate)


def test_creative_lab_uses_finite_slate_and_reports_denominators(monkeypatch) -> None:
    calls: list[str] = []

    def fake_backtest(spec, user_request, panel=None):
        calls.append(spec.name)
        # Only momentum variants pass train; only one survives lockbox.
        if "lockbox" in spec.name.lower():
            verdict = "worth_paper_trading" if "Creative 1" in spec.name else "needs_more_testing"
        else:
            verdict = "worth_paper_trading" if spec.portfolio_family in {
                PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
                PortfolioFamily.DUAL_MOMENTUM,
            } else "needs_more_testing"
        return _state(spec, verdict)

    monkeypatch.setattr(cl, "run_portfolio_backtest", fake_backtest)
    monkeypatch.setattr(cl, "load_portfolio_panel", lambda assets, start, end: None)
    monkeypatch.setattr(
        cl,
        "run_stress_test",
        lambda spec, full_start, full_end, panel=None: {
            "summary": {"verdict": "ROBUST", "overall_confirmed": 8, "overall_runnable": 10, "overall_rate": 0.8},
            "results": [],
        },
    )

    result = cl.run_creative_lab(
        assets=["SPY", "TLT", "GLD", "DBC"],
        start="2010-01-01",
        end="2024-01-01",
        max_candidates=6,
        lockbox_pct=0.25,
    )

    assert result["summary"]["pre_registered_candidates"] == 6
    assert result["summary"]["train_survivors"] == 2
    assert result["summary"]["lockbox_survivors"] == 1
    assert result["summary"]["stress_survivors"] == 1
    assert result["summary"]["verdict"] == "PAPER_TRADE_CANDIDATE"
    assert result["winner"] is not None
    assert len(calls) == 8  # 6 train trials + 2 lockbox retests; no tweak loop.


def test_creative_lab_rejects_if_no_lockbox_survivor(monkeypatch) -> None:
    monkeypatch.setattr(cl, "load_portfolio_panel", lambda assets, start, end: None)
    monkeypatch.setattr(
        cl,
        "run_portfolio_backtest",
        lambda spec, user_request, panel=None: _state(spec, "worth_paper_trading" if "lockbox" not in spec.name else "needs_more_testing"),
    )

    result = cl.run_creative_lab(
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        max_candidates=4,
        lockbox_pct=0.25,
    )

    assert result["winner"] is None
    assert result["summary"]["verdict"] == "REJECTED_NO_LOCKBOX_SURVIVORS"
    assert result["summary"]["stress_survivors"] == 0


def test_creative_lab_mines_anomalies_before_researched_generation(monkeypatch) -> None:
    captured = {}
    researched_spec = PortfolioSpec(
        name="Fact constrained strategy",
        assets=["SPY", "TLT", "GLD"],
        portfolio_family=PortfolioFamily.DUAL_MOMENTUM,
        start_date="2010-01-01",
        end_date="2021-01-01",
        lookback_days=252,
        top_k=1,
        rebalance_days=21,
        hypothesis="Explains mined anomaly.",
    )

    class FakeHypothesis:
        title = "Fact constrained hypothesis"
        mechanism = "Explains the mined fact."
        evidence_to_check = ["fact stability"]
        falsification_tests = ["lockbox fail"]
        portfolio_index = 0

    class FakeResearch:
        research_brief = "Uses anomaly facts."
        hypotheses = [FakeHypothesis()]
        portfolios = [researched_spec]

    fact = AnomalyFact(
        kind="lead_lag",
        leader="GLD",
        follower="SPY",
        lag_days=20,
        score=0.3,
        fact="GLD leads SPY at lag 20.",
        control="Control corr weak.",
    )
    monkeypatch.setattr(cl, "mine_anomalies", lambda assets, start, end, top_n=12: {"facts": [fact]})

    def fake_generate_researched_portfolio_slate(**kwargs):
        captured.update(kwargs)
        return FakeResearch()

    monkeypatch.setattr(cl, "generate_researched_portfolio_slate", fake_generate_researched_portfolio_slate)
    monkeypatch.setattr(cl, "load_portfolio_panel", lambda assets, start, end: None)
    monkeypatch.setattr(cl, "run_portfolio_backtest", lambda spec, user_request, panel=None: _state(spec, "reject"))

    result = cl.run_creative_lab(
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        max_candidates=1,
        lockbox_pct=0.25,
        research_goal="explain anomalies",
    )

    assert captured["anomaly_facts"] == ["GLD leads SPY at lag 20. Control: Control corr weak."]
    assert result["anomaly_facts"] == captured["anomaly_facts"]


def test_creative_lab_can_use_researched_hypothesis_generator(monkeypatch) -> None:
    researched_spec = PortfolioSpec(
        name="Researched macro rotation",
        assets=["SPY", "TLT", "GLD"],
        portfolio_family=PortfolioFamily.DUAL_MOMENTUM,
        start_date="2010-01-01",
        end_date="2021-01-01",
        lookback_days=252,
        top_k=1,
        rebalance_days=21,
        hypothesis="Rates/liquidity regime hypothesis.",
    )

    class FakeHypothesis:
        title = "Rates/liquidity rotation"
        mechanism = "Falling rates can support duration while risk assets de-rate in stress."
        evidence_to_check = ["rate trend", "equity drawdown"]
        falsification_tests = ["lockbox fail"]
        portfolio_index = 0

    class FakeResearch:
        research_brief = "Research brief about rates/liquidity and crisis beta."
        hypotheses = [FakeHypothesis()]
        portfolios = [researched_spec]

    monkeypatch.setattr(cl, "generate_researched_portfolio_slate", lambda **kwargs: FakeResearch())
    monkeypatch.setattr(cl, "mine_anomalies", lambda assets, start, end, top_n=12: {"facts": []})
    monkeypatch.setattr(cl, "load_portfolio_panel", lambda assets, start, end: None)
    monkeypatch.setattr(cl, "run_portfolio_backtest", lambda spec, user_request, panel=None: _state(spec, "needs_more_testing"))

    result = cl.run_creative_lab(
        assets=["SPY", "TLT", "GLD"],
        start="2010-01-01",
        end="2024-01-01",
        max_candidates=1,
        lockbox_pct=0.25,
        research_goal="Find structurally motivated macro strategies.",
    )

    assert result["research_brief"].startswith("Research brief")
    assert result["hypotheses"][0].title == "Rates/liquidity rotation"
    assert result["slate"][0].name == "Researched macro rotation"
    assert result["summary"]["pre_registered_candidates"] == 1
