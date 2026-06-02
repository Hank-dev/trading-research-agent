import pandas as pd

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.workflows import portfolio_research


def make_spec(name: str = "Rot", **overrides) -> PortfolioSpec:
    data = {
        "name": name,
        "assets": ["SPY", "TLT", "GLD"],
        "portfolio_family": PortfolioFamily.CROSS_SECTIONAL_MOMENTUM,
        "start_date": "2015-01-01",
        "end_date": "2023-01-01",
        "lookback_days": 126,
        "top_k": 1,
        "rebalance_days": 21,
        "hypothesis": "Cross-asset momentum rotates into whatever is trending.",
    }
    data.update(overrides)
    return PortfolioSpec(**data)


def fake_result(spec: PortfolioSpec, sharpe: float = 1.0, beats: bool = True) -> BacktestResult:
    return BacktestResult(
        strategy_name=spec.name,
        asset="PORTFOLIO[" + ",".join(spec.assets) + "]",
        start_date=spec.start_date,
        end_date=spec.end_date,
        engine="vectorbt_portfolio",
        metrics=BacktestMetrics(
            total_return_pct=40.0 if beats else 2.0,
            buy_and_hold_return_pct=20.0,
            sharpe_ratio=sharpe,
            max_drawdown_pct=-18.0,
            num_trades=40,
            win_rate_pct=52.0,
            exposure_time_pct=80.0,
            final_equity=14000.0,
            beats_benchmark=beats,
        ),
    )


class FakeBackend:
    def __init__(self, sharpe_by_name: dict[str, float] | None = None):
        self.sharpe_by_name = sharpe_by_name or {}

    def __call__(self):
        return self

    def run(self, spec, panel, aux=None):
        base = spec.name.replace(" (lockbox)", "")
        return fake_result(spec, sharpe=self.sharpe_by_name.get(base, 1.0))


def _patch_backend(monkeypatch, sharpe_by_name=None):
    backend = FakeBackend(sharpe_by_name)
    monkeypatch.setattr(portfolio_research, "PortfolioVectorbtBackend", backend)
    monkeypatch.setattr(
        portfolio_research,
        "load_portfolio_panel",
        lambda assets, start, end: pd.DataFrame({a: [1.0, 2.0] for a in assets}),
    )


def test_run_portfolio_backtest_produces_report_and_metric_checks(monkeypatch) -> None:
    _patch_backend(monkeypatch)
    state = portfolio_research.run_portfolio_backtest(make_spec(), "rotate across assets")

    assert state["critique"].approved is True
    assert state["backtest_result"] is not None
    assert state["report"] is not None
    names = [c.test_name for c in state["backtest_result"].robustness_results]
    assert "Benchmark comparison" in names
    assert "Sharpe ratio significance (PSR)" in names


def test_run_portfolio_backtest_rejects_bad_dates(monkeypatch) -> None:
    _patch_backend(monkeypatch)
    spec = make_spec(start_date="2023-01-01", end_date="2020-01-01")
    state = portfolio_research.run_portfolio_backtest(spec, "x")
    assert state["critique"].approved is False
    assert "backtest_result" not in state


def test_run_portfolio_backtest_handles_load_failure(monkeypatch) -> None:
    monkeypatch.setattr(portfolio_research, "PortfolioVectorbtBackend", FakeBackend())
    monkeypatch.setattr(
        portfolio_research,
        "load_portfolio_panel",
        lambda assets, start, end: (_ for _ in ()).throw(ValueError("no overlap")),
    )
    state = portfolio_research.run_portfolio_backtest(make_spec(), "x")
    assert "backtest_result" not in state
    assert any("no overlap" in e for e in state["errors"])
    assert state["report"].verdict == "error"


def test_run_portfolio_exploration_runs_slate_and_picks_winner(monkeypatch) -> None:
    specs = [make_spec("A"), make_spec("B"), make_spec("C")]
    monkeypatch.setattr(
        portfolio_research, "generate_portfolio_slate", lambda r, n: specs
    )
    _patch_backend(monkeypatch, sharpe_by_name={"A": 0.5, "B": 1.8, "C": 0.9})

    result = portfolio_research.run_portfolio_exploration("rotate", slate_size=3)

    assert len(result["candidates"]) == 3
    assert result["winner_index"] == 1  # B has the highest Sharpe among beaters
    assert "failure_summary" in result
    # DSR check appended to each candidate
    for cand in result["candidates"]:
        names = [c.test_name for c in cand["backtest_result"].robustness_results]
        assert "Deflated Sharpe ratio (DSR)" in names


def test_run_portfolio_exploration_with_lockbox_truncates_and_reruns(monkeypatch) -> None:
    specs = [make_spec("A"), make_spec("B")]
    monkeypatch.setattr(
        portfolio_research, "generate_portfolio_slate", lambda r, n: specs
    )
    seen_ranges: list[tuple[str, str]] = []

    class TrackingBackend(FakeBackend):
        def run(self, spec, panel, aux=None):
            seen_ranges.append((spec.start_date, spec.end_date))
            return super().run(spec, panel, aux)

    monkeypatch.setattr(portfolio_research, "PortfolioVectorbtBackend", TrackingBackend({"A": 1.5, "B": 0.4}))
    monkeypatch.setattr(
        portfolio_research,
        "load_portfolio_panel",
        lambda assets, start, end: pd.DataFrame({a: [1.0, 2.0] for a in assets}),
    )

    result = portfolio_research.run_portfolio_exploration(
        "rotate", slate_size=2, lockbox_pct=0.2
    )

    assert "lockbox" in result
    assert result["lockbox"]["strategy_spec"].name == "A (lockbox)"
    # Slate ran on the truncated range; lockbox ran on the held-out tail.
    slate_end = seen_ranges[0][1]
    assert slate_end < "2023-01-01"
    lockbox_start, lockbox_end = seen_ranges[-1]
    assert lockbox_start > slate_end
    assert lockbox_end == "2023-01-01"


def test_run_portfolio_exploration_handles_slate_failure(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("grok down")

    monkeypatch.setattr(portfolio_research, "generate_portfolio_slate", boom)
    result = portfolio_research.run_portfolio_exploration("x", slate_size=3)
    assert result["candidates"] == []
    assert result["winner_index"] is None
    assert any("grok down" in e for e in result["errors"])


def test_run_portfolio_spec_runs_exact_strategy_without_dsr(monkeypatch) -> None:
    _patch_backend(monkeypatch)

    result = portfolio_research.run_portfolio_spec(make_spec("Exact"), "run exact spec")

    assert len(result["candidates"]) == 1
    assert result["winner_index"] == 0
    assert result["winner_reason"] == "hand_specified_strategy"
    names = [c.test_name for c in result["candidates"][0]["backtest_result"].robustness_results]
    assert "Deflated Sharpe ratio (DSR)" not in names


def test_run_portfolio_spec_with_lockbox_truncates_and_reruns(monkeypatch) -> None:
    seen_ranges: list[tuple[str, str]] = []

    class TrackingBackend(FakeBackend):
        def run(self, spec, panel, aux=None):
            seen_ranges.append((spec.start_date, spec.end_date))
            return super().run(spec, panel, aux)

    monkeypatch.setattr(portfolio_research, "PortfolioVectorbtBackend", TrackingBackend())
    monkeypatch.setattr(
        portfolio_research,
        "load_portfolio_panel",
        lambda assets, start, end: pd.DataFrame({a: [1.0, 2.0] for a in assets}),
    )

    result = portfolio_research.run_portfolio_spec(
        make_spec("Exact"), "run exact spec", lockbox_pct=0.2
    )

    assert "lockbox" in result
    assert result["lockbox"]["strategy_spec"].name == "Exact (lockbox)"
    train_start, train_end = seen_ranges[0]
    lockbox_start, lockbox_end = seen_ranges[-1]
    assert train_start == "2015-01-01"
    assert train_end < "2023-01-01"
    assert lockbox_start > train_end
    assert lockbox_end == "2023-01-01"
