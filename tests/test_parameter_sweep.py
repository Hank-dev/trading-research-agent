import pytest

from trading_research_agent.schemas.backtest import BacktestMetrics, BacktestResult
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategyFamily
from trading_research_agent.workflows import parameter_sweep as ps


def _metrics(sharpe: float, ret: float, bench: float) -> BacktestMetrics:
    return BacktestMetrics(
        total_return_pct=ret,
        buy_and_hold_return_pct=bench,
        sharpe_ratio=sharpe,
        max_drawdown_pct=-12.0,
        num_trades=40,
        win_rate_pct=55.0,
        exposure_time_pct=60.0,
        final_equity=11000.0,
        beats_benchmark=ret > bench,
    )


class FakeGraph:
    """Returns a verdict based on the swept value via a caller-provided rule."""

    def __init__(self, confirm_rule):
        self.confirm_rule = confirm_rule

    def invoke(self, state):
        spec = state["strategy_spec"]
        # Recover the swept value from the spec (entry_window for donchian here).
        value = spec.entry_window
        confirms = self.confirm_rule(value)
        verdict = "worth_paper_trading" if confirms else "needs_more_testing"
        result = BacktestResult(
            strategy_name=spec.name,
            asset=spec.asset,
            start_date=spec.start_date,
            end_date=spec.end_date,
            engine="vectorbt",
            metrics=_metrics(1.2 if confirms else 0.3, 50 if confirms else 5, 20),
        )
        report = ResearchReport(markdown="x", verdict=verdict, reasons=[], next_tests=[])
        return {"strategy_spec": spec, "backtest_result": result, "report": report}


def _patch(monkeypatch, confirm_rule):
    monkeypatch.setattr(ps, "build_research_graph", lambda: FakeGraph(confirm_rule))


def test_plateau_when_contiguous_band_confirms(monkeypatch) -> None:
    # 40,55,70 confirm contiguously; 20 and 120 fail -> PLATEAU.
    _patch(monkeypatch, lambda v: 40 <= v <= 70)
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
        [20, 40, 55, 70, 120], "2015-01-01", "2025-01-01", 0.25,
    )
    assert result["verdict"] == "PLATEAU"
    assert result["plateau_values"] == [40, 55, 70]
    assert result["longest_run"] == 3


def test_spike_when_only_one_value_confirms(monkeypatch) -> None:
    _patch(monkeypatch, lambda v: v == 55)
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
        [20, 40, 55, 70, 120], "2015-01-01", "2025-01-01", 0.25,
    )
    assert result["verdict"] == "SPIKE"
    assert result["longest_run"] == 1


def test_none_when_nothing_confirms(monkeypatch) -> None:
    _patch(monkeypatch, lambda v: False)
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
        [20, 40, 55, 70], "2015-01-01", "2025-01-01", 0.25,
    )
    assert result["verdict"] == "NONE"
    assert result["n_confirmed"] == 0


def test_non_contiguous_confirms_is_spike(monkeypatch) -> None:
    # 20 and 120 confirm but are not adjacent -> longest run 1 -> SPIKE.
    _patch(monkeypatch, lambda v: v in (20, 120))
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
        [20, 40, 55, 70, 120], "2015-01-01", "2025-01-01", 0.25,
    )
    assert result["verdict"] == "SPIKE"
    assert result["n_confirmed"] == 2
    assert result["longest_run"] == 1


def test_rejects_unsweepable_param() -> None:
    with pytest.raises(ValueError, match="not sweepable"):
        ps.run_single_asset_sweep(
            "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "rsi_window",
            [10, 14, 20], "2015-01-01", "2025-01-01", 0.25,
        )


def test_rejects_no_lockbox() -> None:
    with pytest.raises(ValueError, match="lockbox"):
        ps.run_single_asset_sweep(
            "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
            [20, 40, 55], "2015-01-01", "2025-01-01", 0.0,
        )


def test_format_sweep_renders_verdict_and_caveat(monkeypatch) -> None:
    _patch(monkeypatch, lambda v: 40 <= v <= 70)
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.DONCHIAN_BREAKOUT, "entry_window",
        [20, 40, 55, 70, 120], "2015-01-01", "2025-01-01", 0.25,
    )
    text = ps.format_sweep(result)
    assert "PLATEAU" in text
    assert "correlated" in text  # the honesty caveat on sweep DSR
    assert "entry_window" in text


def test_invalid_variant_is_marked(monkeypatch) -> None:
    # slow_window must exceed fast_window; sweeping fast_window above slow (200) is invalid.
    _patch(monkeypatch, lambda v: True)
    result = ps.run_single_asset_sweep(
        "BTC-USD", StrategyFamily.SMA_CROSSOVER, "fast_window",
        [50, 250], "2015-01-01", "2025-01-01", 0.25,
    )
    statuses = {e["value"]: e["status"] for e in result["entries"]}
    assert statuses[250] == "invalid"  # fast 250 > slow 200 -> rejected
    assert statuses[50] == "ok"
