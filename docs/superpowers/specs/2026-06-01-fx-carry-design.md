# Design: `fx_carry` strategy family

Date: 2026-06-01

## Motivation

The accumulated research record falsifies *price-only* rules across six liquid
expressions (macro-asset rotation, US sectors, country equities, commodities,
FX momentum, long-horizon reversal). The one repeatedly-identified live frontier
is **new data**: a genuinely different return driver fed by something other than
price. FX **carry** — long high-interest-rate currencies, short/avoid
low-rate ones — is the textbook carry edge and is *distinct from* FX momentum
(which already failed on the same instruments). It brings a new data input
(short-term interest rates) and a different asset class than the equity/bond/
commodity book, so it is a real orthogonality candidate, not another price rule.

## Data feasibility (verified live)

FRED's keyless `fredgraph.csv` serves OECD 3-month interbank rates for all six
G10 currencies behind the existing FX ETF universe, plus USD:

| ETF | Currency | FRED series        |
|-----|----------|--------------------|
| FXE | EUR (DE) | IR3TIB01DEM156N    |
| FXY | JPY      | IR3TIB01JPM156N    |
| FXB | GBP      | IR3TIB01GBM156N    |
| FXF | CHF      | IR3TIB01CHM156N    |
| FXA | AUD      | IR3TIB01AUM156N    |
| FXC | CAD      | IR3TIB01CAM156N    |
| —   | USD      | IR3TIB01USM156N    |

Series are **monthly, stamped at month-start** (e.g. 2024-12-01 = December's
value), spanning 2005→2025. Carry_i = rate_i − rate_USD.

## Architecture decision — thread an explicit `aux` panel

The signal layer is pure pandas (no IO, look-ahead-testable); preserve that.
Add an optional `aux: pd.DataFrame | None = None` to `compute_target_weights`
carrying a dates×assets **carry panel** (rate differentials, %), loaded upstream
and passed in. Rejected: loading rates inside the signal (breaks purity), and a
separate FX backend (duplicates robustness machinery). Explicit-aux keeps the
signal unit-testable with a hand-built panel and reuses walk-forward / Monte
Carlo / lockbox unchanged.

## Components

1. **Pure transform — `_carry_panel_from_rates(rates_by_asset, usd_rate, target_index, lag_days)`**
   (data_loader): for each monthly series, shift its index forward by `lag_days`
   (availability date), reindex onto `target_index` with forward-fill, then
   `carry[asset] = rate_daily − usd_daily`. No network — unit-testable.
2. **IO wrapper — `load_fx_carry_rates(assets, start, end, target_index)`**: maps
   each asset via a module-level ETF→series dict (case-insensitive; unknown
   symbol → clear error), pulls each series + USD via `load_fred_series`, calls
   the pure transform with `FX_CARRY_PUBLICATION_LAG_DAYS = 60`.
3. **Signal — `_fx_carry_row(close, i, spec, aux)`**: carry score = mean of `aux`
   over `[i − lookback_days + 1, i]` (smooths the monthly print), rank
   **descending**, long the top_k highest-carry equal-weight (`1/top_k`, cash
   remainder). Raises if `aux is None`. Add `aux` to `_target_row` dispatch;
   other families ignore it.
4. **Backend threading** (portfolio_vectorbt): `run` and the walk-forward check
   accept `aux` and slice it to each window via `aux.loc[window.index]`.
5. **Workflow**: load the carry panel (reindexed to the price panel) only when
   `family == fx_carry`; pass `aux=None` for every other family. `paper_trading`
   guarded the same way.
6. **Schema**: `FX_CARRY` enum; reuse `top_k` (validated) and `lookback_days`
   (now the carry-smoothing window); no `skip_recent`. Lag is a module constant.

## Look-ahead discipline

Two stacked guards: (a) the 60-day availability lag baked into the carry panel
(a month-start-stamped rate is invisible until ~2 months later, safely past
publication), and (b) the existing weight `.shift(1)`. A rate change in month m
must not move weights until m + lag.

## Tests (TDD, written first)

- `_carry_panel_from_rates`: differential computed correctly; a value stamped
  month-start is NOT visible until `lag_days` later (look-ahead guard); forward
  fill holds between monthly prints.
- `_fx_carry_row`: longs the highest-carry asset(s) for a hand-built carry panel;
  respects `top_k`; raises when `aux is None`.
- Backend slices `aux` to each walk-forward window without misalignment.
- Schema: `top_k` bounds for `fx_carry`.

## Honest caveat (empirical risk)

Currency ETFs harvest carry imperfectly — some accrue local interest, some don't
track it cleanly. A correct carry *signal* may not yield carry *returns* via
these instruments. The code tests verify signal + lag correctness; whether the
*edge* survives is the pre-registered slate's job. A null result is acceptable
and expected-possible.

## Out of scope (deferred)

Long-short carry (framework is long-only, weights ≤ 1), a cash/absolute-carry
filter variant (a second pre-registered spec, not a code branch), and LLM slate
awareness.

## Success criterion

Code done = tests green, ruff + mypy clean, no new look-ahead. Research payoff
measured by the pre-registered slate: out-of-sample verdict, and correlation of
carry returns with the momentum and reversal books (orthogonality is the point).
