from typing import Literal

from pydantic import BaseModel, Field, model_validator

from trading_research_agent.config import (
    DEFAULT_COMMISSION_PCT,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE_PCT,
)


class CombinedBookSpec(BaseModel):
    """Evaluate whether a hedge overlay improves a core book.

    The core is held continuously (equal-weight across `core_assets`). The overlay
    is a hedge sleeve held at `overlay_weight` either permanently ("static") or
    only when the core's trend turns down ("regime"). The workflow compares the
    combined book against the core held alone over the same window.
    """

    name: str = Field(default="combined_book")
    core_assets: list[str] = Field(description="Core risk assets, held continuously, equal-weight.")
    overlay_assets: list[str] = Field(description="Hedge sleeve assets (e.g. VIXY, GLD, TLT).")
    overlay_weight: float = Field(description="Fraction of the book in the overlay, in (0, 0.5].")
    overlay_rule: Literal["static", "regime"] = Field(
        default="regime",
        description=(
            "static = hold the overlay permanently; regime = hold it only while the "
            "core composite is below its own SMA."
        ),
    )

    start_date: str
    end_date: str

    lookback_days: int = Field(default=100, description="SMA window for the regime trigger.")
    rebalance_days: int = Field(default=21)

    initial_cash: float = Field(default=DEFAULT_INITIAL_CASH, gt=0)
    commission_pct: float = Field(default=DEFAULT_COMMISSION_PCT, ge=0)
    slippage_pct: float = Field(default=DEFAULT_SLIPPAGE_PCT, ge=0)

    @model_validator(mode="after")
    def validate_book(self) -> "CombinedBookSpec":
        core = _dedupe(self.core_assets)
        overlay = _dedupe(self.overlay_assets)
        object.__setattr__(self, "core_assets", core)
        object.__setattr__(self, "overlay_assets", overlay)

        if not core:
            raise ValueError("core_assets must contain at least one asset")
        if not overlay:
            raise ValueError("overlay_assets must contain at least one asset")

        overlap = {a.lower() for a in core} & {a.lower() for a in overlay}
        if overlap:
            raise ValueError(f"core and overlay assets must be disjoint; shared: {overlap}")

        if not 0.0 < self.overlay_weight <= 0.5:
            raise ValueError("overlay_weight must be in (0, 0.5]")
        if self.lookback_days < 20 or self.lookback_days > 504:
            raise ValueError("lookback_days must be between 20 and 504")
        if self.rebalance_days < 1 or self.rebalance_days > 252:
            raise ValueError("rebalance_days must be between 1 and 252")
        return self

    def all_assets(self) -> list[str]:
        return [*self.core_assets, *self.overlay_assets]


def _dedupe(assets: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in assets:
        a = raw.strip()
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out
