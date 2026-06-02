# Design: `filtered_rsi_mean_reversion` strategy family

Date: 2026-06-01

## Motivation

Two findings set this up:
1. Plain RSI mean reversion has a **real, repeatable out-of-sample tilt on non-bull,
   choppy assets** (international equity indices: 75% of trials beat buy-and-hold,
   median held-out Sharpe ~0.53) — but it does NOT clear the full verdict bar
   (significance / walk-forward / Monte Carlo). It is a lead, not a tradeable edge.
2. The popular claim is that mean reversion "works once you add a volatility/regime
   filter." When we tested the breakout analogue (`filtered_donchian_breakout`),
   filters only reduced participation without conjuring edge — but there was no
   baseline edge to improve. Here there IS one, so the claim is finally well-posed.

Commodities (DBC/USO/GLD/SLV/DBA/DBB) are a good habitat: choppy, mean-reverting,
survivorship-bias-free, no secular bull. The question: **does a calm-volatility
filter improve mean reversion, or just trade less?**

## Signal

Mirrors the existing `filtered_donchian_breakout` shape, inverted for mean
reversion (calm volatility instead of expansion; no trend filter):

```
rsi      = rsi(close, rsi_window)
atr_now  = atr(high, low, close, atr_window)
calm_vol = atr_now < sma(atr_now, atr_ma_window)   # volatility NOT spiking
entries  = (rsi < oversold_threshold) & calm_vol    # buy oversold dips only when calm
exits    = rsi > exit_threshold                      # exit on mean reversion; vol gates ENTRY only
```

Reuses existing `StrategySpec` params: `rsi_window`, `oversold_threshold`,
`exit_threshold`, `atr_window`, `atr_ma_window`. `regime_window` is NOT used
(vol-only filter — fewest knobs).

## Changes (small; mirror the filtered-breakout work)

1. **schemas/strategy.py** — add `FILTERED_RSI_MEAN_REVERSION` enum; validation
   requires `rsi_window`, `oversold_threshold`, `exit_threshold`, `atr_window`,
   `atr_ma_window`.
2. **backtesting/backends/vectorbt_backend.py** — one `_signals` branch.
3. **workflows/parameter_sweep.py** — `_FAMILY_DEFAULTS`
   (`rsi_window 14, oversold 30, exit 50, atr_window 14, atr_ma_window 20`) and
   `_SWEEPABLE` (`rsi_window, oversold_threshold, exit_threshold, atr_window,
   atr_ma_window`).
4. **cli/args.py** — add `filtered_rsi_mean_reversion` to `--sweep-family` choices.

## Tests (TDD, written first)

- Calm-vol gating: a constructed series with two oversold (RSI < threshold)
  episodes — one calm (low ATR), one a vol spike (high ATR) — produces an entry
  only in the calm episode.
- Filtered entries are a strict subset of plain RSI entries (the filter only
  removes, never adds).
- Schema: the family requires its 5 params (omitting one raises).
- `parameter_sweep` exposes the family in defaults and sweepable params.

## The experiment (pre-registered)

Run BOTH `rsi_mean_reversion` (baseline) and `filtered_rsi_mean_reversion` on the
SAME commodity basket, OOS lockbox, sweeping `oversold_threshold` ∈ {25,30,35}:

- Basket: **DBC, USO, GLD, SLV, DBA, DBB** (broad, oil, gold, silver, agriculture,
  base metals). 2010–2024, lockbox 0.25.
- Compare median held-out Sharpe and beat-buy-and-hold rate, **filtered vs plain**.

Decision rule (declared before running):
- Filter **helps** if filtered median held-out Sharpe is materially higher than
  plain on the same basket.
- Filter is **just participation reduction** (the breakout lesson) if Sharpe is
  similar/lower with fewer trades.
- A null/negative result is fully acceptable and expected-possible.

## Out of scope (deferred)

Trend/ranging regime filter (chose vol-only), ADX indicator, a cross-sectional
commodity MR portfolio, LLM/graph awareness beyond the deterministic sweep path.

## Success criterion

Code done = tests green, ruff + mypy clean. Research payoff = the honest
filtered-vs-plain comparison, whichever way it lands.
