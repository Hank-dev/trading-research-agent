from trading_research_agent.tools import trial_budget as tb


def trial(asset="PORTFOLIO[SPY,TLT,GLD]", sharpe=0.8, **extra):
    rec = {
        "asset": asset,
        "is_lockbox": False,
        "metrics": {"sharpe_ratio": sharpe},
    }
    rec.update(extra)
    return rec


def lockbox(asset="PORTFOLIO[SPY,TLT,GLD]", sharpe=1.0, verdict="worth_paper_trading", **extra):
    rec = {
        "asset": asset,
        "is_lockbox": True,
        "verdict": verdict,
        "metrics": {"sharpe_ratio": sharpe},
        "start_date": "2021-01-01",
        "end_date": "2024-01-01",
    }
    rec.update(extra)
    return rec


def test_no_portfolio_research():
    b = tb.assess_trial_budget([])
    assert b["verdict"] == "NO_PORTFOLIO_RESEARCH_YET"
    assert b["n_trials"] == 0


def test_counts_only_portfolio_non_lockbox_non_stress_as_trials():
    records = [
        trial(),
        trial(),
        lockbox(),  # not a trial
        {"asset": "PORTFOLIO[X,Y]", "mode": "stress", "is_lockbox": False, "verdict": "ROBUST"},  # not a trial
        {"asset": "SPY", "is_lockbox": False, "metrics": {"sharpe_ratio": 0.5}},  # single-asset, not portfolio
    ]
    b = tb.assess_trial_budget(records)
    assert b["n_trials"] == 2
    assert b["n_lockbox"] == 1


def test_nothing_confirmed_verdict():
    records = [trial(), trial(), lockbox(verdict="needs_more_testing")]
    b = tb.assess_trial_budget(records)
    assert b["verdict"] == "NOTHING_CONFIRMED"
    assert b["n_confirmed"] == 0


def test_confirmed_but_fails_after_correction_with_many_noisy_trials():
    # Many trials with a wide spread of Sharpes -> high deflation threshold.
    records = [trial(sharpe=s) for s in [1.5, -0.5, 0.9, 0.2, 1.1, -0.2, 0.7, 1.3, 0.1, 0.6]]
    # The confirmed candidate's held-out Sharpe is modest relative to that spread.
    records.append(lockbox(sharpe=0.9))
    b = tb.assess_trial_budget(records)
    assert b["n_confirmed"] == 1
    assert b["candidates"][0]["cross_run_dsr"] < 0.95
    assert b["verdict"] == "CONFIRMED_BUT_NOT_AFTER_CORRECTION"


def test_survives_multiple_testing_with_strong_long_sample_candidate():
    # Few, tight trials and a strong candidate with a long held-out window.
    records = [trial(sharpe=0.7), trial(sharpe=0.75)]
    records.append(lockbox(sharpe=3.0, start_date="2010-01-01", end_date="2024-01-01"))
    b = tb.assess_trial_budget(records)
    assert b["candidates"][0]["cross_run_dsr"] >= 0.95
    assert b["verdict"] == "SURVIVES_MULTIPLE_TESTING"


def test_format_budget_reports_counts_and_verdict():
    records = [trial(), trial(), lockbox(verdict="needs_more_testing")]
    text = tb.format_budget(tb.assess_trial_budget(records))
    assert "Portfolio shots taken" in text
    assert "NOTHING_CONFIRMED" in text


def test_format_budget_empty():
    text = tb.format_budget(tb.assess_trial_budget([]))
    assert "No portfolio trials" in text
