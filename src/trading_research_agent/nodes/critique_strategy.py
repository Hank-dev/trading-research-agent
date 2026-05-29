from datetime import date

from pydantic import ValidationError

from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.schemas.strategy import BacktestEngine, StrategySpec


CRYPTO_PREFIXES = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "AVAX"}


def critique_strategy_node(state: dict) -> dict:
    spec = state.get("strategy_spec")
    problems: list[str] = []
    warnings: list[str] = []
    required_changes: list[str] = []

    if spec is None:
        problems.append("No valid StrategySpec is available for critique.")
        required_changes.append("Parse the request into a valid StrategySpec.")
        return {
            "critique": StrategyCritique(
                approved=False,
                problems=problems,
                warnings=warnings,
                required_changes=required_changes,
            )
        }

    if not isinstance(spec, StrategySpec):
        try:
            spec = StrategySpec.model_validate(spec)
        except ValidationError as exc:
            problems.append(f"StrategySpec validation failed: {exc}")
            required_changes.append("Fix invalid strategy fields before backtesting.")
            return {
                "critique": StrategyCritique(
                    approved=False,
                    problems=problems,
                    warnings=warnings,
                    required_changes=required_changes,
                )
            }

    _check_dates(spec, problems, required_changes)
    _check_required_fields(spec, problems, required_changes)
    _add_warnings(spec, state.get("user_request", ""), warnings)

    return {
        "critique": StrategyCritique(
            approved=not problems,
            problems=problems,
            warnings=warnings,
            required_changes=required_changes,
        )
    }


def _check_dates(
    spec: StrategySpec, problems: list[str], required_changes: list[str]
) -> None:
    try:
        start = date.fromisoformat(spec.start_date)
        end = date.fromisoformat(spec.end_date)
    except ValueError as exc:
        problems.append(f"Invalid ISO date: {exc}")
        required_changes.append("Use YYYY-MM-DD dates.")
        return

    if end <= start:
        problems.append("End date must be after start date.")
        required_changes.append("Choose an end date after the start date.")


def _check_required_fields(
    spec: StrategySpec, problems: list[str], required_changes: list[str]
) -> None:
    if spec.timeframe != "1d":
        problems.append("MVP only supports daily timeframe: 1d.")
    if spec.backtest_engine == BacktestEngine.NAUTILUS:
        problems.append("Nautilus backend is not implemented.")
    if spec.commission_pct is None or spec.commission_pct < 0:
        problems.append("Commission must be present and non-negative.")
    if spec.slippage_pct is None or spec.slippage_pct < 0:
        problems.append("Slippage must be present and non-negative.")
    if not spec.benchmark:
        problems.append("Benchmark must be present.")
    if not spec.asset:
        problems.append("Asset must be present.")
    if not spec.hypothesis:
        problems.append("Hypothesis must be present.")

    if problems:
        required_changes.append("Fix deterministic critique failures before backtesting.")


def _add_warnings(spec: StrategySpec, user_request: str, warnings: list[str]) -> None:
    asset_prefix = spec.asset.split("-")[0].upper()
    if asset_prefix in CRYPTO_PREFIXES:
        warnings.append("Asset appears to be crypto; market structure can change quickly.")

    try:
        start = date.fromisoformat(spec.start_date)
        end = date.fromisoformat(spec.end_date)
        if (end - start).days < 365 * 3:
            warnings.append("Backtest period is less than 3 years.")
    except ValueError:
        pass

    warnings.append("Strategy is tested on only one asset.")

    lowered = user_request.lower()
    if "drawdown" in lowered and ("guarantee" in lowered or "guaranteed" in lowered):
        warnings.append("User requested drawdown certainty; backtests cannot guarantee drawdown.")
    if "best" in lowered or "profitable" in lowered:
        warnings.append("User used outcome-seeking language without objective criteria.")
