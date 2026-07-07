import numpy as np
import pandas as pd

from trading_research_agent.workflows import event_followthrough as ef


def _panel_with_delayed_uup_btc_effect() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    idx = pd.date_range("2017-01-01", periods=900, freq="B")
    uup_rets = pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)
    # Alternating dollar weakness shocks create enough events for train/lockbox.
    for start in range(120, len(idx), 90):
        uup_rets.iloc[start : start + 20] -= 0.006
    uup = 100 * (1 + uup_rets).cumprod()

    z = ef._rolling_event_zscore(uup, event_window=20, z_lookback_days=252)
    btc_rets = pd.Series(rng.normal(-0.004, 0.006, len(idx)), index=idx)
    event = z <= -1.0
    active = ef._position_from_events(event, lag_days=40, hold_days=20)
    btc_rets.loc[active.astype(bool)] += 0.026
    btc = 100 * (1 + btc_rets).cumprod()
    return pd.DataFrame({"UUP": uup, "BTC-USD": btc}, index=idx)


def test_event_followthrough_position_waits_then_holds() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    event = pd.Series(False, index=idx)
    event.iloc[2] = True

    pos = ef._position_from_events(event, lag_days=3, hold_days=4)

    assert pos.iloc[0:5].sum() == 0
    assert pos.iloc[5:9].sum() == 4
    assert pos.iloc[9:].sum() == 0


def test_generate_slate_converts_negative_event_spread_to_weakness(monkeypatch) -> None:
    fact = ef.AnomalyFact(
        kind="event_followthrough",
        leader="UUP",
        follower="BTC-USD",
        lag_days=40,
        score=10.0,
        fact="x",
        control="x",
        train_score=-5.0,
        holdout_score=-6.0,
    )
    monkeypatch.setattr(
        ef,
        "mine_anomalies",
        lambda assets, start, end, top_n: {"facts": [fact]},
    )

    slate = ef.generate_event_followthrough_slate(["UUP", "BTC-USD"], "2017-01-01", "2020-01-01")

    assert len(slate) == 1
    assert slate[0].leader == "UUP"
    assert slate[0].target == "BTC-USD"
    assert slate[0].direction == "weakness"
    assert slate[0].lag_days == 40


def test_event_followthrough_lab_promotes_robust_synthetic_rule(monkeypatch) -> None:
    panel = _panel_with_delayed_uup_btc_effect()
    fact = ef.AnomalyFact(
        kind="event_followthrough",
        leader="UUP",
        follower="BTC-USD",
        lag_days=40,
        score=10.0,
        fact="UUP weakness leads BTC",
        control="synthetic",
        train_score=-5.0,
        holdout_score=-5.0,
    )
    monkeypatch.setattr(
        ef,
        "mine_anomalies",
        lambda assets, start, end, top_n: {"facts": [fact]},
    )

    result = ef.run_event_followthrough_lab(
        ["UUP", "BTC-USD"],
        "2017-01-01",
        "2020-06-30",
        max_candidates=1,
        lockbox_pct=0.25,
        panel=panel,
    )

    assert result["summary"]["pre_registered_candidates"] == 1
    assert result["summary"]["train_survivors"] == 1
    assert result["summary"]["lockbox_survivors"] == 1
    assert result["results"][0]["lockbox"].sharpe is not None
    records = ef.learning_records_from_event_followthrough(result)
    assert len(records) == 1
    assert records[0]["learning_status"] in {"winner", "fragile_lockbox_survivor"}
    assert records[0]["params"]["leader"] == "UUP"
    assert records[0]["params"]["target"] == "BTC-USD"
    assert records[0]["lesson"]
    assert records[0]["gate_denominators"]["pre_registered_candidates"] == 1
    assert "UUP weakness" in ef.format_event_followthrough_lab(result)
