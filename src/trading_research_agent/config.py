import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    api_key: str
    model: str
    base_url: str | None = None


def load_settings() -> Settings:
    load_dotenv()
    provider = _resolve_provider()
    if provider == "xai":
        return _load_xai_settings()
    if provider == "openai":
        return _load_openai_settings()
    raise RuntimeError("Unsupported LLM_PROVIDER. Use 'xai' or 'openai'.")


def _resolve_provider() -> str:
    configured_provider = os.getenv("LLM_PROVIDER")
    if configured_provider:
        return configured_provider.strip().lower()
    if os.getenv("XAI_API_KEY"):
        return "xai"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "xai"


def _load_xai_settings() -> Settings:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY. Add it to your .env file.")
    return Settings(
        llm_provider="xai",
        api_key=api_key,
        model=os.getenv("XAI_MODEL", "grok-4.3"),
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    )


def _load_openai_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")
    return Settings(
        llm_provider="openai",
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    )
