# Trading Research Agent

## What This Is

A local research tool for converting natural-language trading ideas into structured, reproducible backtests.

## What This Is Not

This is not a live trading bot and does not place trades.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For VPS deployment, Docker Compose, systemd, and nginx examples are in
[docs/VPS_DEPLOYMENT.md](docs/VPS_DEPLOYMENT.md).

## Environment Setup

Copy `.env.example` to `.env` and set your xAI API key:

```bash
LLM_PROVIDER=xai
XAI_API_KEY=your_xai_api_key_here
XAI_MODEL=grok-4.3
XAI_BASE_URL=https://api.x.ai/v1
```

The parser uses Grok through xAI's OpenAI-compatible Chat Completions endpoint.
OpenAI can still be used by setting `LLM_PROVIDER=openai` and `OPENAI_API_KEY`.

## Example Usage

```bash
trade-research --save-report "Create a BTC SMA crossover strategy from 2018-01-01 to 2025-12-31"
```

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for a flowchart of the agent workflow.

### Forward paper-trading (the only ungameable test)

A strategy can be tuned and re-tested against history until it looks good, but it cannot be fit to time that has not happened yet. This mode records a confirmed strategy as a dated paper position and replays it forward using real data accrued since inception.

```bash
trade-research --paper-trade                      # open a position on the latest confirmed winner
trade-research --paper-trade --inception 2025-01-01   # or pin the inception date
trade-research --paper-status                     # replay forward, realized vs backtested
```

Inception defaults to the backtest's end date — the exact boundary past which every bar is out of sample. `--paper-status` reports realized return / annualized / drawdown since inception against the backtested expectation, with a read of `TOO_EARLY`, `TRACKING`, `DIVERGING`, or `DRAWDOWN_BREACH`. Re-run it over weeks and months; the longer the forward record, the more it can confirm or break a strategy that nothing else can.

For a persistent paper book with cash, positions, and a daily ledger:

```bash
trade-research --live-open                        # open a stateful paper book
trade-research --live-tick                        # advance every open book to latest data
trade-research --live-tick --book-id abc123 --as-of 2026-06-19
trade-research --live-status                      # inspect NAV/cash/positions
trade-research --live-auto-promote                # open a new book for a newer confirmed winner
```

These `--live-*` commands still do **not** place broker orders. They persist a stateful paper-trading book under `outputs/live_books/` so repeated ticks are idempotent and auditable.

### Combined-book evaluation (the honest way to judge a hedge)

A hedge usually loses money in isolation, so a standalone backtest answers the wrong question. This mode answers the right one: does a small overlay improve the *combined* book versus holding the core alone?

```bash
# VIX overlay, only carried when the core trend turns down
trade-research --combined-book --core SPY --overlay VIXY \
  --overlay-weight 0.1 --overlay-rule regime --start 2012-01-01 --end 2024-01-01

# Diversifier overlay held permanently
trade-research --combined-book --core SPY --overlay GLD,TLT \
  --overlay-weight 0.2 --overlay-rule static --start 2012-01-01 --end 2024-01-01
```

It runs the combined book and the core alone over the same window and compares return, drawdown, and Sharpe. Verdict:

- `IMPROVES_RISK_ADJUSTED` — combined Sharpe is strictly higher (a genuine free improvement; rare).
- `REDUCES_DRAWDOWN_AT_COST` — Sharpe not higher, but drawdown materially shallower (insurance you pay for).
- `NOT_WORTH_IT` — dragged returns without enough drawdown benefit.

`--overlay-rule` is `static` (hold the overlay always) or `regime` (hold it only while the core composite is below its own SMA — avoids bleeding on long-vol ETFs in calm markets).

Add `--lockbox-pct 0.25` to confirm the overlay's benefit on a held-out tail the comparison never saw. The benefit is only `CONFIRMED OUT OF SAMPLE` if it persists there — a diversification edge that worked for a decade can still be a regime artifact (e.g. the 40-year bond bull market that broke in 2022).

### Robustness stress-testing a confirmed winner

Once a portfolio strategy has confirmed on a held-out lockbox, stress-test it — the goal is to *break* it, not improve it:

```bash
trade-research --stress            # deterministic battery
trade-research --stress --suggest  # + a Grok plain-English read of the fragility
```

It reconstructs the most recent lockbox-confirmed portfolio winner from `outputs/history.jsonl` and re-checks, for each perturbation, whether it *still* confirms on a held-out segment:

- **Lockbox sensitivity** — different held-out cut points (0.15 / 0.20 / 0.25 / 0.30). If it only works at one cut, the original pass was a lucky window.
- **Parameter neighbors** — nearby lookback / rebalance / top_k. If only the exact settings work, it is overfit.
- **Leave-one-out universe** — drop each asset in turn. If dropping one asset destroys the edge, it was a bet on that asset, not a portfolio effect.

Verdict is `ROBUST` (survives its neighborhood — the strongest evidence this toolkit produces; next step is forward paper trading), `FRAGILE` (knife-edge — distrust it, and do NOT tune it to fix the failures), or `BROKEN` (does not even reconfirm at baseline). This mode is deterministic; `--suggest` only narrates the numbers.

### Multi-asset portfolio strategies

Research rotation strategies that move capital *across* a universe of assets, rather than trading one asset in isolation. A Grok research-director pre-registers a slate of structurally-distinct portfolio rules, all backtested as a single cash-shared portfolio against an equal-weight benchmark.

```bash
trade-research --portfolio --explore 4 --lockbox-pct 0.2 \
  "rotate across stocks, bonds and commodities from 2010-01-01 to 2025-01-01"
```

Run one exact hand-specified portfolio without asking Grok to generate a slate:

```bash
trade-research --portfolio-spec \
  --assets GLD,SPY,USO,TLT,BTC-USD \
  --family cross_sectional_momentum \
  --lookback 126 --top-k 2 --rebalance 21 \
  --start 2015-01-01 --end 2026-05-29 \
  --lockbox-pct 0.2
```

Run several exact universes from a file:

```bash
trade-research --portfolio-batch examples/portfolio_batch.json --lockbox-pct 0.2
```

Volatility-scaled momentum is useful for mixed universes that include high-volatility
assets such as BTC or oil:

```bash
trade-research --portfolio-spec \
  --assets GLD,SPY,USO,TLT,BTC-USD \
  --family volatility_scaled_momentum \
  --lookback 126 --rebalance 21 \
  --start 2015-01-01 --end 2026-05-29 \
  --lockbox-pct 0.2
```

Batch files can be JSON or YAML:

```json
{
  "defaults": {
    "family": "cross_sectional_momentum",
    "lookback": 126,
    "top_k": 2,
    "rebalance": 21,
    "start": "2010-01-01",
    "end": "2025-01-01"
  },
  "portfolios": [
    {"name": "Classic cross asset", "assets": ["SPY", "TLT", "DBC", "GLD"]},
    {"name": "Equity rates gold", "assets": "SPY,QQQ,TLT,IEF,GLD"}
  ]
}
```

Supported portfolio families:

- `cross_sectional_momentum` — hold the top-K assets by trailing return, rebalanced periodically.
- `dual_momentum` — cross-sectional momentum with an absolute filter: only hold assets whose own trailing return is positive, otherwise sit in cash (Antonacci-style defense).
- `equal_weight_trend` — equal-weight each asset, but only while it is above its own moving average.
- `time_series_momentum` — hold each asset's equal-weight slice only while its own trailing return is positive.
- `volatility_scaled_momentum` — hold assets with positive trailing return, sized by inverse recent volatility so high-volatility assets do not dominate.
- `crisis_hedge` — two-asset core/volatility-hedge rule with `--hedge-weight`.

Notes:

- `--portfolio` requires `--explore N` (the slate size). `--lockbox-pct` works the same as in single-asset explore: the winner is re-tested on a held-out tail the slate never saw.
- `--portfolio-spec` is deterministic: it uses `--assets`, `--family`, `--lookback`, `--top-k`, `--rebalance`, `--start` and `--end` exactly as provided. A lockbox-confirmed result is logged in the same portfolio history format, so `--stress` and `--paper-trade` can pick it up.
- The director only proposes *structurally distinct* hypotheses (different families / universes / lookback regimes), never parameter tweaks of a strategy already in the slate.
- Every per-asset price series is auto-routed: BTC via Coin Metrics, Nasdaq via FRED, everything else via Tiingo (so `TIINGO_API_KEY` must be set for most universes). If any asset in a universe cannot be loaded, that whole candidate is skipped — they must share overlapping history.
- Results flow into the same `outputs/history.jsonl`, tagged `mode=portfolio`, so `--history --suggest` sees them alongside single-asset trials.
- Completed research runs automatically refresh `outputs/dashboard.html`; `--report-html` is still available when you want to rebuild it manually.
- To inspect a past run without rerunning data, use `trade-research --history-detail RUN_ID` or a query such as `trade-research --history-detail "SPY, TLT, DBC, GLD"`.

Preflight data availability before spending a backtest run:

```bash
trade-research --data-health \
  --assets SPY,TLT,DBC,GLD \
  --start 2010-01-01 --end 2025-01-01
```

This reports each asset's data source, Tiingo cache coverage, row count, and whether the common aligned panel is runnable. If Tiingo quota/auth fails, later uncached Tiingo assets are skipped instead of repeatedly hitting a doomed API call.

### Most automated: campaign mode across an asset universe

Run the same idea as a pre-registered slate against every asset in a fixed universe in one command. Each asset gets its own slate + lockbox, then a cross-asset summary is printed.

```bash
trade-research --campaign --universe SPY,QQQ,IWM,BTC-USD,GLD \
  --explore 3 --lockbox-pct 0.2 \
  "trend following from 2015-01-01 to 2025-01-01"
```

- `--campaign` activates campaign mode. Requires `--universe` and `--explore`.
- `--universe` is a comma-separated list of asset symbols. Duplicates and case variants are collapsed.
- Every per-asset slate writes to the same `outputs/history.jsonl`, so `--history --suggest` afterwards reads the campaign in aggregate.
- The campaign summary explicitly tells you the cumulative trial count and reminds you that per-slate DSR does not deflate against the whole campaign — this is by design, not a bug, but you should internalize it.

### Recommended: pre-registered slate with held-out lockbox

Generate N distinct strategies up front, backtest all, and re-test the winner once on a held-out tail of the date range:

```bash
trade-research --explore 5 --lockbox-pct 0.2 \
  "BTC trend following from 2020-01-01 to 2025-01-15"
```

- `--explore N` pre-registers N strategies before any backtest runs, removing the feedback loop that lets the model optimize against the verdict function.
- `--lockbox-pct X` reserves the trailing fraction `X` (e.g. `0.2` = last 20%) of the date range. The slate never sees it; only the winner is re-run on it. The final verdict on the lockbox segment is the one to trust.
- `--explore` reports a deflated Sharpe ratio per candidate that accounts for the number of trials, pushing back against the multiple-testing bias the slate introduces.

### Deprecated: iterative refinement

> **Deprecated.** `--iterate-once` and `--iterate-until-pass` repeatedly tweak a strategy in response to its own backtest results, which is exactly the gradient-descent-against-the-verdict pattern that produces over-fit research. Prefer `--explore --lockbox-pct` for new work. These flags remain for now but will be removed in a future release.

```bash
trade-research --iterate-once "BTC SMA crossover from 2020-01-01 to 2025-01-15"
trade-research --iterate-until-pass --max-iterations 5 "BTC SMA crossover from 2020-01-01 to 2025-01-15"
```

## Supported Strategies

- SMA crossover
- Donchian breakout
- RSI mean reversion

## Market Data

- BTC data uses Coin Metrics community API `PriceUSD`, matching the market monitor.
- Nasdaq/QQQ requests use FRED `NASDAQ100` close-only daily data.
- US equities and ETFs (including gold ETFs `GLD`, `IAU`) use **Tiingo** when `TIINGO_API_KEY` is set in `.env` — recommended over yfinance because Yahoo blocks non-browser HTTP clients and `curl_cffi`'s OpenSSL workaround is unreliable on some Linux distros.
- If you see `curl: (35) TLS connect error` or `possibly delisted; no timezone found` from yfinance for normal ETFs like `SPY`, `TLT`, or `GLD`, treat it as a Yahoo/yfinance transport failure. Set `TIINGO_API_KEY` in `.env` and rerun.
- Without `TIINGO_API_KEY`, the pipeline falls back to `yfinance`, then to Stooq if `STOOQ_API_KEY` is configured.
- Get a free Tiingo API key at https://www.tiingo.com/ (500 requests/day on the free tier, plenty for backtesting).

Tiingo responses are cached locally under `outputs/cache/tiingo/` by symbol and covered date range. Re-running the same strategy, or a narrower date range already covered by a previous fetch, reuses cached rows without spending Tiingo quota. Delete `outputs/cache/tiingo/` to force a fresh download, or set `TRADING_RESEARCH_DISABLE_CACHE=1` to bypass the cache for a run. On a VPS, set `TRADING_RESEARCH_CACHE_DIR` to a persistent path such as `/var/lib/trading-research-agent/cache`.

## Backtesting Engine Choice

The default research backend is `vectorbt`, which enables fast strategy runs plus walk-forward and Monte Carlo robustness checks. `backtesting.py` remains available behind the same backend interface.

## Output Files

Equity charts, history, paper positions, the HTML dashboard, and optional Markdown reports are written to `outputs/` by default. Set `TRADING_RESEARCH_OUTPUT_DIR` to move these files to a persistent VPS path.

## Limitations

- Daily bars only — no intraday data or execution modeling.
- Long-only, no leverage, no shorting.
- Backtests are vectorized approximations: fills at the close, fixed
  fee/slippage assumptions, no order-book or market-impact modeling.
- Data coverage depends on the configured providers (Tiingo / Coin Metrics /
  FRED / yfinance); universes must share overlapping history.
- This is a research tool. Nothing here places trades or constitutes advice.

## Roadmap

The original MVP roadmap (walk-forward, parameter sensitivity, multi-asset, and
paper-trading) is now implemented — see `--stress`, `--portfolio`/`--campaign`,
and `--paper-trade`. Possible next directions:

- Refactor the `app.py` CLI surface into a dedicated `cli/` package.
- Centralize backtest assumptions (fees, slippage, lockbox) in `config.py`.
- Campaign-wide deflation across all trials, not just per-slate.
- Longer forward paper-trading track records to confirm `ROBUST` winners.
