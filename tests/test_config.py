import pytest

from trading_research_agent import config


def clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "LLM_PROVIDER",
        "XAI_API_KEY",
        "XAI_MODEL",
        "XAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)


def test_defaults_to_xai_and_requires_xai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_llm_env(monkeypatch)

    with pytest.raises(RuntimeError, match="Missing XAI_API_KEY"):
        config.load_settings()


def test_loads_xai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")

    settings = config.load_settings()

    assert settings.llm_provider == "xai"
    assert settings.api_key == "test-xai-key"
    assert settings.model == "grok-4.3"
    assert settings.base_url == "https://api.x.ai/v1"


def test_loads_openai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    settings = config.load_settings()

    assert settings.llm_provider == "openai"
    assert settings.api_key == "test-openai-key"
    assert settings.model == "gpt-4.1-mini"
    assert settings.base_url is None
