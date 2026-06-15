import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Backtest assumptions: single source of truth ---------------------------
# Every spec (single-asset, portfolio, combined-book) defaults to these unless
# explicitly overridden. Keep them here, not duplicated per schema, so a change
# to the assumed cost model applies everywhere and cannot silently drift.
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_COMMISSION_PCT = 0.001   # 0.1% per trade
DEFAULT_SLIPPAGE_PCT = 0.0005    # 0.05% per trade

# Trailing fraction of the date range reserved as a held-out lockbox for the
# out-of-sample confirmation gate (used by robustness stress-testing).
DEFAULT_LOCKBOX_PCT = 0.20
DEFAULT_OUTPUT_DIR = "outputs"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    api_key: str
    model: str
    base_url: str | None = None


def get_output_dir() -> Path:
    configured = os.getenv("TRADING_RESEARCH_OUTPUT_DIR", DEFAULT_OUTPUT_DIR).strip()
    return Path(configured or DEFAULT_OUTPUT_DIR)


def get_output_path(*parts: str) -> Path:
    return get_output_dir().joinpath(*parts)


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
