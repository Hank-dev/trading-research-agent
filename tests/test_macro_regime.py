import numpy as np
import pandas as pd

from trading_research_agent.workflows import macro_regime as mr


def make_panel(rows: int = 160) -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", periods=rows)
    return pd.DataFrame({a: np.arange(rows, dtype=float) for a in mr.ASSETS}, index=idx)


def weekly_signal(values: list[float]) -> pd.Series:
    idx = pd.date_range("2014-09-01", periods=len(values), freq="W")
    return pd.Series(values, index=idx, dtype=float)


_CFG = {"publication_lag_days": 7, "change_window_days": 5}


# ---- regime computation ----


def test_rising_balance_sheet_is_expansion() -> None:
    panel = make_panel()
    signal = weekly_signal(list(range(60)))  # strictly increasing
    exp = mr._regime_for_panel(signal, panel.index, _CFG)
    assert exp.iloc[20:].all()  # after warmup, always expansion


def test_falling_balance_sheet_is_contraction() -> None:
    panel = make_panel()
    signal = weekly_signal(list(range(60, 0, -1)))  # strictly decreasing
    exp = mr._regime_for_panel(signal, panel.index, _CFG)
    assert not exp.iloc[20:].any()


def test_no_lookahead_future_spike_does_not_leak() -> None:
    panel = make_panel()
    flat = weekly_signal([100.0] * 60)
    base = mr._regime_for_panel(flat, panel.index, _CFG)
    spiked = flat.copy()
    spiked.iloc[-1] = 1_000_000.0  # a spike far in the future relative to early dates
    after = mr._regime_for_panel(spiked, panel.index, _CFG)
    # Early regime values must be identical — the future spike cannot leak back.
    assert (base.iloc[:60].to_numpy() == after.iloc[:60].to_numpy()).all()


def test_regime_has_no_nan_and_matches_panel_length() -> None:
    panel = make_panel()
    exp = mr._regime_for_panel(weekly_signal([100.0] * 60), panel.index, _CFG)
    assert len(exp) == len(panel)
    assert not exp.isna().any()


# ---- weight mapping (frozen) ----


def test_expansion_weights_are_equal_risk_basket() -> None:
    panel = make_panel()
    expansion = pd.Series(True, index=panel.index)
    w = mr._build_weights(panel, expansion, change_window=5)
    row = w.dropna(how="all").iloc[0]
    for a in mr.EXPANSION_ASSETS:
        assert row[a] == 0.25
    assert row["TLT"] == 0.0


def test_contraction_weights_are_all_tlt() -> None:
    panel = make_panel()
    expansion = pd.Series(False, index=panel.index)
    w = mr._build_weights(panel, expansion, change_window=5)
    row = w.dropna(how="all").iloc[0]
    assert row["TLT"] == 1.0
    for a in mr.EXPANSION_ASSETS:
        assert row[a] == 0.0


def test_weights_execute_next_bar() -> None:
    # The first change_window rows must carry no orders (shifted, warmup).
    panel = make_panel()
    expansion = pd.Series(True, index=panel.index)
    w = mr._build_weights(panel, expansion, change_window=5)
    assert w.iloc[:5].isna().all().all()
