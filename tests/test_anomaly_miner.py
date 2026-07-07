import numpy as np
import pandas as pd

from trading_research_agent.workflows import anomaly_miner as am


def _panel_with_lagged_relationship() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    leader_rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    follower_rets = leader_rets.shift(5).fillna(0) + pd.Series(
        rng.normal(0, 0.002, len(idx)), index=idx
    )
    unrelated_rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    return pd.DataFrame(
        {
            "LEADER": 100 * (1 + leader_rets).cumprod(),
            "FOLLOWER": 100 * (1 + follower_rets).cumprod(),
            "NOISE": 100 * (1 + unrelated_rets).cumprod(),
        },
        index=idx,
    )


def test_lead_lag_miner_finds_specific_leader_follower_fact() -> None:
    facts = am.mine_lead_lag_anomalies(
        _panel_with_lagged_relationship(),
        lags=(1, 5, 10),
        horizon=1,
        min_abs_corr=0.35,
        top_n=5,
    )

    top = facts[0]
    assert top.kind == "lead_lag"
    assert top.leader == "LEADER"
    assert top.follower == "FOLLOWER"
    assert top.lag_days == 5
    assert top.score > 0.8
    assert "LEADER" in top.fact
    assert "FOLLOWER" in top.fact
    assert "lag 5" in top.fact


def test_anomaly_miner_reports_regime_conditioned_asymmetry() -> None:
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    signal = pd.Series(np.r_[np.linspace(100, 150, 130), np.linspace(150, 100, 130)], index=idx)
    target_returns = pd.Series(np.r_[np.full(130, 0.002), np.full(130, -0.001)], index=idx)
    panel = pd.DataFrame(
        {
            "SIGNAL": signal,
            "TARGET": 100 * (1 + target_returns).cumprod(),
        },
        index=idx,
    )

    facts = am.mine_regime_anomalies(panel, trend_window=20, forward_horizon=5, min_spread_pct=0.8)

    fact = next(f for f in facts if f.leader == "SIGNAL" and f.follower == "TARGET")
    assert fact.kind == "regime_split"
    assert fact.score > 0
    assert "up-trend" in fact.fact
    assert "down-trend" in fact.fact


def test_event_followthrough_miner_finds_conditional_spread() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-01", periods=700, freq="B")
    leader_rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    leader_price = 100 * (1 + leader_rets).cumprod()

    event_return = leader_price.pct_change(20)
    z = (event_return - event_return.rolling(252, min_periods=80).mean()) / event_return.rolling(
        252, min_periods=80
    ).std()
    follower_rets = pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)
    follower_rets.loc[z.shift(5) >= 1.0] += 0.006
    follower_rets.loc[z.shift(5) <= -1.0] -= 0.006
    panel = pd.DataFrame(
        {
            "LEADER": leader_price,
            "FOLLOWER": 100 * (1 + follower_rets).cumprod(),
        },
        index=idx,
    )

    facts = am.mine_event_followthrough_anomalies(
        panel,
        lags=(5,),
        horizon=5,
        min_events=10,
        min_spread_pct=1.0,
        top_n=3,
    )

    fact = next(f for f in facts if f.leader == "LEADER" and f.follower == "FOLLOWER")
    assert fact.kind == "event_followthrough"
    assert fact.lag_days == 5
    assert fact.train_score is not None and fact.train_score > 0
    assert fact.holdout_score is not None and fact.holdout_score > 0
    assert fact.adjusted_p_value is not None


def test_mine_anomalies_combines_and_sorts_facts(monkeypatch) -> None:
    panel = _panel_with_lagged_relationship()
    monkeypatch.setattr(am, "load_portfolio_panel", lambda assets, start, end: panel)

    result = am.mine_anomalies(
        assets=["LEADER", "FOLLOWER", "NOISE"],
        start="2020-01-01",
        end="2020-12-31",
        top_n=3,
    )

    assert result["assets"] == ["LEADER", "FOLLOWER", "NOISE"]
    assert result["tests_scanned"] > 0
    assert result["facts"]
    assert result["facts"][0].score >= result["facts"][-1].score
    assert "lead_lag" in {fact.kind for fact in result["facts"]}


def test_format_anomalies_includes_denominators_and_facts() -> None:
    fact = am.AnomalyFact(
        kind="lead_lag",
        leader="GLD",
        follower="NZDUSD",
        lag_days=20,
        score=0.31,
        fact="GLD 20d returns lead NZDUSD 20d returns at lag 20 with corr +0.31.",
        control="First-half corr was +0.04.",
    )
    text = am.format_anomaly_report(
        {"assets": ["GLD", "NZDUSD"], "start": "2010-01-01", "end": "2024-12-31", "facts": [fact]}
    )

    assert "Assets: GLD, NZDUSD" in text
    assert "Facts reported: 1" in text
    assert "GLD 20d returns lead NZDUSD" in text
    assert "First-half corr" in text
