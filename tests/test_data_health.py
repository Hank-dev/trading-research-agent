import pandas as pd

from trading_research_agent.workflows import data_health


def make_frame(rows: int = 301) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": [1.0] * rows,
            "High": [1.0] * rows,
            "Low": [1.0] * rows,
            "Close": [1.0] * rows,
            "Volume": [100.0] * rows,
        },
        index=index,
    )


def test_check_data_health_reports_runnable_panel(monkeypatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(data_health, "tiingo_cache_status", lambda *a, **k: {"enabled": True, "covered": True})
    monkeypatch.setattr(
        data_health,
        "load_ohlcv_for_asset",
        lambda asset, start, end: make_frame(),
    )

    result = data_health.check_data_health(["SPY", "TLT"], "2020-01-01", "2020-10-27")

    assert result["runnable"] is True
    assert result["common_rows"] == 301
    assert [check.status for check in result["checks"]] == ["ok", "ok"]
    assert [check.cache for check in result["checks"]] == ["covered", "covered"]


def test_check_data_health_skips_later_uncached_tiingo_after_quota_error(monkeypatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    cache_by_asset = {
        "SPY": {"enabled": True, "covered": False},
        "TLT": {"enabled": True, "covered": False},
    }
    calls: list[str] = []

    monkeypatch.setattr(
        data_health,
        "tiingo_cache_status",
        lambda asset, start, end: cache_by_asset[asset],
    )

    def fake_load(asset, start, end):
        calls.append(asset)
        raise ValueError("Tiingo failed and fallback was not attempted: HTTP 429")

    monkeypatch.setattr(data_health, "load_ohlcv_for_asset", fake_load)

    result = data_health.check_data_health(["SPY", "TLT"], "2020-01-01", "2020-10-27")

    assert result["runnable"] is False
    assert calls == ["SPY"]
    assert result["checks"][0].status == "error"
    assert result["checks"][1].status == "error"
    assert "Skipped after Tiingo quota/auth failure" in result["checks"][1].detail


def test_check_data_health_reports_common_overlap_too_short(monkeypatch) -> None:
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(data_health, "tiingo_cache_status", lambda *a, **k: {"enabled": True, "covered": True})

    def fake_load(asset, start, end):
        frame = make_frame()
        if asset == "TLT":
            frame = frame.iloc[50:]
        return frame

    monkeypatch.setattr(data_health, "load_ohlcv_for_asset", fake_load)

    result = data_health.check_data_health(["SPY", "TLT"], "2020-01-01", "2020-10-27")

    assert result["runnable"] is False
    assert result["common_rows"] == 251
    assert "Only 251 common aligned rows" in result["reason"]
