"""Campaign mode: run the same idea across a fixed universe of assets.

For each asset in the universe, runs one `--explore N` slate with the same
lockbox split. Returns a per-asset breakdown plus a cross-asset aggregate.

Deliberately *not* a parameter search. Each asset still gets a slate of
genuinely distinct hypotheses, not parameter tweaks of one — so the campaign
is exploring "do these strategy families work on these assets" rather than
"find the best parameter combo."
"""

from collections import Counter
from typing import TypedDict

from trading_research_agent.workflows.explore_research import ExploreResult, run_exploration


class AssetCampaignSlot(TypedDict, total=False):
    asset: str
    exploration: ExploreResult
    pass_count: int
    lockbox_verdict: str | None
    error: str


class CampaignSummary(TypedDict):
    total_trials: int
    total_lockbox_runs: int
    assets_with_any_pass: list[str]
    assets_with_lockbox_pass: list[str]
    by_verdict: dict[str, int]
    failed_checks: dict[str, int]


class CampaignResult(TypedDict, total=False):
    idea: str
    universe: list[str]
    slate_size: int
    lockbox_pct: float
    slots: list[AssetCampaignSlot]
    summary: CampaignSummary


def run_campaign(
    idea: str,
    universe: list[str],
    slate_size: int,
    lockbox_pct: float = 0.0,
) -> CampaignResult:
    if slate_size < 1:
        raise ValueError("slate_size must be >= 1")
    if not universe:
        raise ValueError("universe must contain at least one asset")

    seen: set[str] = set()
    deduped_universe: list[str] = []
    for raw in universe:
        asset = raw.strip()
        if not asset or asset.lower() in seen:
            continue
        seen.add(asset.lower())
        deduped_universe.append(asset)

    slots: list[AssetCampaignSlot] = []
    for asset in deduped_universe:
        slots.append(_run_slot(idea, asset, slate_size, lockbox_pct))

    return {
        "idea": idea,
        "universe": deduped_universe,
        "slate_size": slate_size,
        "lockbox_pct": lockbox_pct,
        "slots": slots,
        "summary": _build_summary(slots),
    }


def _run_slot(
    idea: str,
    asset: str,
    slate_size: int,
    lockbox_pct: float,
) -> AssetCampaignSlot:
    per_asset_request = f"{idea} on {asset}"
    try:
        exploration = run_exploration(
            per_asset_request,
            slate_size=slate_size,
            lockbox_pct=lockbox_pct,
        )
    except Exception as exc:
        return {"asset": asset, "error": f"Exploration failed: {exc}"}

    pass_count = sum(
        1
        for candidate in exploration.get("candidates", [])
        if _verdict_of(candidate) == "worth_paper_trading"
    )
    lockbox_verdict = _verdict_of(exploration.get("lockbox")) if exploration.get("lockbox") else None

    return {
        "asset": asset,
        "exploration": exploration,
        "pass_count": pass_count,
        "lockbox_verdict": lockbox_verdict,
    }


def _verdict_of(state: dict | None) -> str | None:
    if state is None:
        return None
    report = state.get("report")
    if report is None:
        return None
    return report.verdict


def _build_summary(slots: list[AssetCampaignSlot]) -> CampaignSummary:
    total_trials = 0
    total_lockbox = 0
    assets_with_any_pass: list[str] = []
    assets_with_lockbox_pass: list[str] = []
    verdict_counts: Counter[str] = Counter()
    failed_check_counts: Counter[str] = Counter()

    for slot in slots:
        exploration = slot.get("exploration")
        if exploration is None:
            continue
        candidates = exploration.get("candidates", [])
        total_trials += len(candidates)
        if exploration.get("lockbox") is not None:
            total_lockbox += 1

        if slot.get("pass_count", 0) > 0:
            assets_with_any_pass.append(slot["asset"])
        if slot.get("lockbox_verdict") == "worth_paper_trading":
            assets_with_lockbox_pass.append(slot["asset"])

        failure_summary = exploration.get("failure_summary")
        if failure_summary is not None:
            for verdict, count in failure_summary.get("verdict_counts", {}).items():
                verdict_counts[verdict] += count
            for check, count in failure_summary.get("failed_check_counts", {}).items():
                failed_check_counts[check] += count

    return {
        "total_trials": total_trials,
        "total_lockbox_runs": total_lockbox,
        "assets_with_any_pass": assets_with_any_pass,
        "assets_with_lockbox_pass": assets_with_lockbox_pass,
        "by_verdict": dict(verdict_counts.most_common()),
        "failed_checks": dict(failed_check_counts.most_common()),
    }
