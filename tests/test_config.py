from pathlib import Path

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
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_BASE_URL",
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


def test_loads_deepseek_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    settings = config.load_settings()

    assert settings.llm_provider == "deepseek"
    assert settings.api_key == "test-deepseek-key"
    assert settings.model == "deepseek-chat"
    assert settings.base_url == "https://api.deepseek.com/v1"


def test_output_path_uses_configured_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    output_dir = tmp_path / "research-output"
    monkeypatch.setenv("TRADING_RESEARCH_OUTPUT_DIR", str(output_dir))

    assert config.get_output_dir() == output_dir
    assert config.get_output_path("history.jsonl") == output_dir / "history.jsonl"


def test_output_path_falls_back_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_RESEARCH_OUTPUT_DIR", "")

    assert config.get_output_dir() == Path("outputs")
