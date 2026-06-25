"""Research-driven strategy hypothesis generation.

This module is upstream of backtesting. Its job is to turn a broad research goal
into a small pre-registered slate of falsifiable portfolio hypotheses. It must not
look at backtest results, mutate candidates after results, or optimize knobs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from trading_research_agent.config import load_settings
from trading_research_agent.schemas.portfolio import PortfolioSpec

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_FALLBACK_MODEL = "accounts/fireworks/models/deepseek-v4-pro"

RESEARCH_PROMPT = """You are a systematic trading research director.

Your task is NOT to find a profitable backtest. Your task is to research and pre-register a small slate of falsifiable, structurally distinct portfolio hypotheses BEFORE any backtest result exists.

Think like a skeptical macro/systematic researcher:
- Prefer structural mechanisms over chart-pattern guesses.
- Examples of structural mechanisms: inflation shocks, rates/liquidity regimes, crisis beta, term premia, commodity terms-of-trade, trend persistence, long-horizon reversal, volatility scaling, defensive de-risking.
- For every hypothesis, state what evidence should be checked and what would falsify it.
- Convert each hypothesis into one PortfolioSpec using only supported portfolio families.
- If empirical anomaly facts are supplied, every hypothesis must explicitly explain
  at least one supplied fact. Do not ignore the facts and fall back to generic
  momentum / mean-reversion claims.

Supported families:
- cross_sectional_momentum
- dual_momentum
- equal_weight_trend
- time_series_momentum
- volatility_scaled_momentum
- cross_sectional_reversal
- fx_carry only if explicitly researching currency carry and suitable ETF/rate data exists
- crisis_hedge only for exactly two assets [core risk asset, long-volatility hedge] with hedge_weight in (0, 0.5]

Hard constraints:
- Produce exactly the requested number of portfolios unless validation makes one impossible.
- Use only the supplied asset universe. Do not invent tickers.
- Every portfolio must use the supplied start_date and end_date.
- Daily data only, vectorbt engine only.
- Each portfolio must map to at least one hypothesis by zero-based portfolio_index.
- Do not claim a strategy works. Say what would falsify it.
- Do not create near-duplicates that differ only by small parameter tweaks.
- This is a pre-registration manifest: it cannot be revised after testing.
"""


class ResearchHypothesis(BaseModel):
    title: str = Field(description="Short title for the structural hypothesis")
    mechanism: str = Field(description="Why this effect might exist structurally")
    evidence_to_check: list[str] = Field(
        min_length=1,
        description="Observable non-result facts or diagnostics worth checking before trusting the idea",
    )
    falsification_tests: list[str] = Field(
        min_length=1,
        description="Tests that would make us reject the hypothesis",
    )
    portfolio_index: int = Field(
        ge=0,
        description="Zero-based index of the PortfolioSpec that operationalizes this hypothesis",
    )


class ResearchedPortfolioSlate(BaseModel):
    research_brief: str = Field(
        description="Concise synthesis of the researched market mechanisms behind the slate"
    )
    hypotheses: list[ResearchHypothesis] = Field(min_length=1)
    portfolios: list[PortfolioSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mapping(self) -> "ResearchedPortfolioSlate":
        max_index = len(self.portfolios) - 1
        mapped = {h.portfolio_index for h in self.hypotheses}
        if any(i > max_index for i in mapped):
            raise ValueError("hypothesis portfolio_index points past portfolios list")
        missing = set(range(len(self.portfolios))) - mapped
        if missing:
            raise ValueError(f"every portfolio needs at least one hypothesis; missing {sorted(missing)}")
        return self


def generate_researched_portfolio_slate(
    user_request: str,
    assets: list[str],
    start: str,
    end: str,
    slate_size: int,
    anomaly_facts: list[str] | None = None,
) -> ResearchedPortfolioSlate:
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1")
    clean_assets = _clean_assets(assets)
    if len(clean_assets) < 2:
        raise ValueError("research hypothesis generation needs at least 2 assets")

    kwargs = {
        "user_request": user_request,
        "assets": clean_assets,
        "start": start,
        "end": end,
        "slate_size": slate_size,
        "anomaly_facts": anomaly_facts or [],
    }
    # Fallback chain: primary LangChain LLM -> Fireworks DeepSeek V4 Pro -> Codex CLI
    fallbacks = [
        ("primary LLM", _invoke_research_model),
        ("Fireworks DeepSeek V4 Pro", _invoke_fireworks_model),
        ("Codex CLI", _invoke_codex_model),
    ]
    errors: list[str] = []
    for name, callable_fn in fallbacks:
        try:
            raw = callable_fn(**kwargs)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        slate = _validate_slate(raw)
        normalized = [_normalize_spec(spec, clean_assets, start, end) for spec in slate.portfolios[:slate_size]]
        hypotheses = [h for h in slate.hypotheses if h.portfolio_index < len(normalized)]
        result = ResearchedPortfolioSlate(
            research_brief=slate.research_brief,
            hypotheses=hypotheses,
            portfolios=normalized,
        )
        if len(result.portfolios) != min(slate_size, len(slate.portfolios)):
            raise ValueError("researched slate normalization changed portfolio count unexpectedly")
        return result

    raise RuntimeError(
        "Researched slate generation failed with all fallbacks. "
        + "; ".join(errors)
    )


def _validate_slate(raw: object) -> ResearchedPortfolioSlate:
    if isinstance(raw, ResearchedPortfolioSlate):
        return raw
    return ResearchedPortfolioSlate.model_validate(raw)


def _invoke_research_model(
    user_request: str,
    assets: list[str],
    start: str,
    end: str,
    slate_size: int,
    anomaly_facts: list[str] | None = None,
) -> ResearchedPortfolioSlate:
    settings = load_settings()
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=settings.model,
        temperature=0.2,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    structured_model = model.with_structured_output(ResearchedPortfolioSlate)
    return structured_model.invoke(
        [
            SystemMessage(content=RESEARCH_PROMPT),
            HumanMessage(
                content=(
                    f"Research goal: {user_request}\n"
                    f"Allowed assets: {', '.join(assets)}\n"
                    f"Date range: {start} to {end}\n"
                    f"Requested portfolios: {slate_size}\n\n"
                    f"Empirical anomaly facts to explain:\n{_format_anomaly_facts(anomaly_facts or [])}\n\n"
                    "First write the research brief and hypotheses, then map each hypothesis to a PortfolioSpec."
                )
            ),
        ]
    )


def _invoke_fireworks_model(
    user_request: str,
    assets: list[str],
    start: str,
    end: str,
    slate_size: int,
    anomaly_facts: list[str] | None = None,
) -> ResearchedPortfolioSlate:
    api_key = os.getenv("FIREWORKS_FALLBACK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Fireworks DeepSeek V4 Pro fallback requires FIREWORKS_FALLBACK_API_KEY "
            "(set it in .env or export it)"
        )
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=FIREWORKS_FALLBACK_MODEL,
        temperature=0.2,
        api_key=api_key,
        base_url=FIREWORKS_BASE_URL,
    )
    structured_model = model.with_structured_output(ResearchedPortfolioSlate)
    return structured_model.invoke(
        [
            SystemMessage(content=RESEARCH_PROMPT),
            HumanMessage(
                content=(
                    f"Research goal: {user_request}\n"
                    f"Allowed assets: {', '.join(assets)}\n"
                    f"Date range: {start} to {end}\n"
                    f"Requested portfolios: {slate_size}\n\n"
                    f"Empirical anomaly facts to explain:\n{_format_anomaly_facts(anomaly_facts or [])}\n\n"
                    "First write the research brief and hypotheses, then map each hypothesis to a PortfolioSpec."
                )
            ),
        ]
    )


def _invoke_codex_model(
    user_request: str,
    assets: list[str],
    start: str,
    end: str,
    slate_size: int,
    anomaly_facts: list[str] | None = None,
) -> ResearchedPortfolioSlate:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex CLI not found on PATH")

    timeout = int(os.getenv("TRADING_RESEARCH_CODEX_TIMEOUT", "300"))
    prompt = _build_codex_prompt(user_request, assets, start, end, slate_size, anomaly_facts or [])
    schema = ResearchedPortfolioSlate.model_json_schema()

    with tempfile.TemporaryDirectory(prefix="trade-research-codex-") as tmp:
        tmpdir = Path(tmp)
        schema_path = tmpdir / "researched_portfolio_slate.schema.json"
        output_path = tmpdir / "codex_last_message.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")

        cmd = [
            codex,
            "exec",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=_codex_subprocess_env(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"codex exec failed with exit {completed.returncode}: {_summarize_codex_error(detail)}"
            )
        if not output_path.exists():
            raise RuntimeError("codex exec produced no --output-last-message file")
        return _parse_codex_json(output_path.read_text(encoding="utf-8"))


def _summarize_codex_error(detail: str) -> str:
    lowered = detail.lower()
    auth_markers = (
        "missing bearer",
        "no codex credentials",
        "401 unauthorized",
        "authentication",
    )
    if any(marker in lowered for marker in auth_markers):
        return (
            "Codex CLI is installed but not authenticated for standalone use. "
            "Run `codex login` (or configure a supported Codex/OpenAI auth env var), "
            "then retry --research-slate."
        )
    return detail[:800]


def _codex_subprocess_env() -> dict[str, str]:
    """Run Codex with its own CLI auth, not the app's broken LLM env vars."""
    env = os.environ.copy()
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "XAI_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "FIREWORKS_FALLBACK_API_KEY",
        "LLM_PROVIDER",
    ):
        env.pop(key, None)
    env.setdefault("NO_COLOR", "1")
    return env


def _build_codex_prompt(
    user_request: str,
    assets: list[str],
    start: str,
    end: str,
    slate_size: int,
    anomaly_facts: list[str],
) -> str:
    return (
        RESEARCH_PROMPT
        + "\n\nReturn ONLY valid JSON matching the provided output schema. No Markdown.\n\n"
        + f"Research goal: {user_request}\n"
        + f"Allowed assets: {', '.join(assets)}\n"
        + f"Date range: {start} to {end}\n"
        + f"Requested portfolios: {slate_size}\n\n"
        + f"Empirical anomaly facts to explain:\n{_format_anomaly_facts(anomaly_facts)}\n\n"
        + "First write the research brief and hypotheses, then map each hypothesis to a PortfolioSpec."
    )


def _parse_codex_json(text: str) -> ResearchedPortfolioSlate:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    return ResearchedPortfolioSlate.model_validate(payload)


def _format_anomaly_facts(anomaly_facts: list[str]) -> str:
    if not anomaly_facts:
        return "- None supplied. Be extra specific about the structural mechanism and falsification tests."
    return "\n".join(f"- {fact}" for fact in anomaly_facts)


def _normalize_spec(
    spec: PortfolioSpec,
    assets: list[str],
    start: str,
    end: str,
) -> PortfolioSpec:
    """Freeze data scope to the user-supplied preregistration boundary."""
    return spec.model_copy(
        update={
            "assets": assets,
            "start_date": start,
            "end_date": end,
        }
    )


def _clean_assets(assets: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in assets:
        asset = raw.strip()
        if asset and asset.lower() not in seen:
            seen.add(asset.lower())
            out.append(asset)
    return out
