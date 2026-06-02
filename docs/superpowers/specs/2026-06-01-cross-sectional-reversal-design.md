# Design: `cross_sectional_reversal` strategy family

Date: 2026-06-01

## Motivation

The research history (161 logged backtests) is almost entirely one economic
edge: trend/momentum (cross-sectional, dual, TSMOM, vol-scaled, SMA, Donchian).
The only non-momentum idea present is single-asset RSI mean-reversion, which
barely registers (best Sharpe 0.11). Because the tool re-deflates every shot
against the cumulative cross-run trial budget, piling on more momentum variants
makes the multiple-testing hurdle *worse*, not better — the momentum vein is
mined out.

The honest way to add value is a **genuinely orthogonal edge**, so the prize is
not standalone Sharpe but **low return-correlation** with the existing momentum
survivors: an uncorrelated sleeve at Sharpe ~0.4 can lift a *combined* book past
a DSR bar that no single momentum sleeve clears.

The first such edge is **long-horizon cross-sectional reversal** (De Bondt &
Thaler, 1985): assets that underperformed over a multi-year window tend to
mean-revert, an effect that is empirically distinct from (and often
anti-correlated with) 6–12-month momentum.

## The load-bearing design point

Naive reversal — "rank by trailing return, buy the losers" over the same 6–12mo
window momentum uses — is just *inverted momentum*: negatively correlated with
the existing book, not orthogonal to it. What makes long-term reversal an
independent edge is the **skip-recent gap**: measure the return over a long
window but *exclude the most recent ~12 months* (months 13–60 mean-revert; the
last 12 trend). This requires a new `skip_recent_days` parameter — it is the
part that is not "momentum with a minus sign."

## Signal: `_reversal_row`

At each rebalance bar `i`:

```
past   = close.iloc[i - lookback_days]      # e.g. ~5y ago
recent = close.iloc[i - skip_recent_days]   # e.g. ~12mo ago  (NOT close.iloc[i])
score  = recent / past - 1.0
```

Rank **ascending**, hold the bottom `top_k` (most beaten-down over the gapped
window), equal-weight `1 / top_k` (mirrors `_momentum_row` weighting, including
the divide-by-`top_k`-not-`len(selected)` convention so any shortfall stays in
cash). No absolute filter.

Look-ahead is already handled by the shared pipeline: `compute_target_weights`
`.shift(1)`s every row, and `_rebalance_row_indices` starts at `lookback_days`,
so `i - lookback_days >= 0` always holds. No change to either.

## Changes (all small, all reuse existing patterns)

1. **schemas/portfolio.py**
   - Add `CROSS_SECTIONAL_REVERSAL = "cross_sectional_reversal"` to
     `PortfolioFamily`.
   - Add `skip_recent_days: int = Field(default=252, ...)`.
   - Raise the `lookback_days` ceiling from 504 to 1260 (~5y).
   - Validation: `skip_recent_days >= 21` and `skip_recent_days < lookback_days`;
     add `CROSS_SECTIONAL_REVERSAL` to the existing `top_k` validation tuple.

2. **tools/portfolio_signals.py**
   - Add `_reversal_row(close, i, spec)` and one dispatch line in `_target_row`.

3. **workflows/portfolio_batch.py**
   - Pass `skip_recent_days` through in `_spec_from_dict` (alias `skip_recent`),
     default 252.

## Invocation

Deterministic batch path: `--portfolio-batch <file>.json` with explicit,
pre-registered specs — no LLM, which is the intended discipline.

**Out of scope (deferred):** teaching `generate_portfolio_slate` about the
family, and critique/report copy. The family is fully usable without those; the
slate prompt already advertises only a subset of existing families.

## Tests (TDD — written first)

- `_reversal_row` picks the worst *gapped-window* performer, and **ignores** a
  recent-window rally (panel where an asset crashed 5y→1y ago but rallied in the
  last 12mo → still selected).
- The selected weight row lands one bar after its decision bar (no look-ahead).
- Schema: `skip_recent_days < lookback_days` enforced; `lookback_days=1260`
  accepted, `1261` rejected; `top_k` bounds apply to reversal.
- Batch loader round-trips `skip_recent_days` and its alias.

## Runtime guidance (not code)

A 5y lookback + 1y skip needs ~6y of history before the first trade. Run on
long-history universes (SPY/QQQ/EFA/EEM/TLT/IEF/GLD/DBC all predate 2007) and
expect the tradeable window to start ~6y after the youngest asset's inception.

## Success criterion

Code "done" = tests green, ruff + mypy clean. **Research payoff** is separate
and measured after a pre-registered permutation slate: low return-correlation
with the momentum survivors, and whether a combined book clears cross-run DSR.
Standalone reversal underperforming momentum is an acceptable, expected outcome
— orthogonality is the point.
