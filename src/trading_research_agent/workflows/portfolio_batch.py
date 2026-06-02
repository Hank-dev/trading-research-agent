"""Batch runner for exact portfolio specifications.

This is the deterministic counterpart to LLM-generated portfolio slates: a file
contains the universes and parameters, and the runner executes exactly those
specs with the normal portfolio pipeline and optional lockbox split.
"""

from pathlib import Path
import json
from typing import Any

from trading_research_agent.schemas.portfolio import PortfolioFamily, PortfolioSpec
from trading_research_agent.workflows.portfolio_research import run_portfolio_spec


def load_portfolio_batch(path: str | Path) -> list[PortfolioSpec]:
    """Load portfolio specs from a JSON or YAML file.

    Supported shapes:

    - a top-level list of portfolio objects
    - an object with `defaults` and `portfolios`

    Common aliases are accepted so batch files can use concise CLI-style keys:
    `family`, `lookback`, `rebalance`, `start`, and `end`.
    """
    source = Path(path)
    data = _read_structured_file(source)

    defaults: dict[str, Any] = {}
    if isinstance(data, dict):
        defaults = dict(data.get("defaults") or {})
        items = data.get("portfolios")
    else:
        items = data

    if not isinstance(items, list) or not items:
        raise ValueError("portfolio batch must contain a non-empty `portfolios` list")

    specs: list[PortfolioSpec] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"portfolio #{index} must be an object")
        merged = {**defaults, **item}
        specs.append(_spec_from_dict(merged, index=index))
    return specs


def run_portfolio_batch(
    path: str | Path,
    *,
    lockbox_pct: float = 0.0,
) -> dict[str, Any]:
    specs = load_portfolio_batch(path)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, spec in enumerate(specs, start=1):
        user_request = _request_for_spec(spec, index=index)
        try:
            result = run_portfolio_spec(spec, user_request, lockbox_pct=lockbox_pct)
        except Exception as exc:
            errors.append(f"portfolio #{index} ({spec.name}) failed: {exc}")
            continue
        results.append(
            {
                "index": index,
                "spec": spec,
                "user_request": user_request,
                "result": result,
            }
        )

    return {
        "path": str(path),
        "lockbox_pct": lockbox_pct,
        "count": len(specs),
        "results": results,
        "errors": errors,
    }


def _read_structured_file(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"portfolio batch file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared.
            raise ValueError("YAML batch files require PyYAML; use JSON instead") from exc
        return yaml.safe_load(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("Unknown batch extension; use .json, .yaml, or .yml") from exc
        return yaml.safe_load(text)


def _spec_from_dict(raw: dict[str, Any], *, index: int) -> PortfolioSpec:
    assets = _assets(raw.get("assets"))
    family_raw = _get(raw, "portfolio_family", "family", default=None)
    if family_raw is None:
        raise ValueError(f"portfolio #{index} requires `family` or `portfolio_family`")
    family = PortfolioFamily(family_raw)

    start = _get(raw, "start_date", "start", default=None)
    end = _get(raw, "end_date", "end", default=None)
    if not start or not end:
        raise ValueError(f"portfolio #{index} requires `start`/`end` dates")

    name = raw.get("name") or _default_name(family, assets)
    hypothesis = raw.get("hypothesis") or _default_hypothesis(family, assets)

    return PortfolioSpec(
        name=str(name),
        assets=assets,
        portfolio_family=family,
        start_date=str(start),
        end_date=str(end),
        lookback_days=int(_get(raw, "lookback_days", "lookback", default=126)),
        top_k=int(_get(raw, "top_k", "top-k", default=1)),
        rebalance_days=int(_get(raw, "rebalance_days", "rebalance", default=21)),
        skip_recent_days=int(_get(raw, "skip_recent_days", "skip_recent", default=252)),
        hedge_weight=_optional_float(raw.get("hedge_weight")),
        hypothesis=str(hypothesis),
    )


def _get(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _assets(value: Any) -> list[str]:
    if isinstance(value, str):
        assets = [s for s in (part.strip() for part in value.split(",")) if s]
    elif isinstance(value, list):
        assets = [str(s).strip() for s in value if str(s).strip()]
    else:
        raise ValueError("each portfolio requires `assets` as a list or comma string")
    if len(assets) < 2:
        raise ValueError("each portfolio must include at least two assets")
    return assets


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _default_name(family: PortfolioFamily, assets: list[str]) -> str:
    return f"{family.value.replace('_', ' ').title()} ({', '.join(assets)})"


def _default_hypothesis(family: PortfolioFamily, assets: list[str]) -> str:
    return (
        f"Pre-registered {family.value} portfolio across {', '.join(assets)} "
        "from a deterministic batch file."
    )


def _request_for_spec(spec: PortfolioSpec, *, index: int) -> str:
    return (
        f"Batch portfolio #{index}: {spec.portfolio_family.value} across "
        f"{', '.join(spec.assets)} from {spec.start_date} to {spec.end_date}; "
        f"lookback={spec.lookback_days}, top_k={spec.top_k}, "
        f"rebalance={spec.rebalance_days}."
    )
