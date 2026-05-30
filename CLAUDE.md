# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local research tool that turns trading ideas into structured, reproducible backtests. It is **not** a live trading bot and never places trades. Its defining purpose is *anti-overfitting discipline* — see "Design philosophy" below, which is load-bearing: most of the architecture exists to keep a researcher from fooling themselves, and features that undermine that (e.g. "iterate until the backtest passes") are deliberately deprecated.

## Commands

```bash
pip install -e ".[dev]"        # install (editable, with pytest/ruff/mypy)
python -m pytest               # run all tests (~230 tests)
python -m pytest tests/test_trial_budget.py::test_nothing_confirmed_verdict   # single test
ruff check .                   # lint
mypy src                       # type check
```

Run the CLI via the module form (the `trade-research` console script may not be on `~/.local/bin` PATH):

```bash
python -m trading_research_agent.app --portfolio --explore 4 --lockbox-pct 0.2 "rotate across stocks, bonds, commodities 2010-01-01 to 2025-01-01"
```

Requires `.env` (copy `.env.example`): `LLM_PROVIDER=xai` + `XAI_API_KEY` (Grok via xAI's OpenAI-compatible endpoint), or `LLM_PROVIDER=openai` + `OPENAI_API_KEY`. `TIINGO_API_KEY` is **load-bearing** — most non-BTC/non-Nasdaq universes route through Tiingo and fail without it.

vectorbt notes: backends and vectorbt tests set `NUMBA_DISABLE_JIT=1` (required under Python 3.14); vectorbt-dependent tests guard with `pytest.importorskip("vectorbt")`.

## Architecture

The CLI ([app.py](src/trading_research_agent/app.py)) is a flat dispatch over mutually-exclusive mode flags. `main()` calls `load_dotenv()` first so even non-LLM modes (e.g. `--stress`) see `TIINGO_API_KEY`. Arg parsing lives in [cli/args.py](src/trading_research_agent/cli/args.py), terminal rendering in [cli/render.py](src/trading_research_agent/cli/render.py), and history/dashboard side-effects in [cli/history_io.py](src/trading_research_agent/cli/history_io.py).

There are **two parallel research pipelines**:

- **Single-asset** — a LangGraph in [workflows/research_graph.py](src/trading_research_agent/workflows/research_graph.py): `parse_strategy → critique_strategy → (run_backtest → robustness_checks) → generate_report`, with a conditional edge that skips the backtest if critique rejects. Invoking the graph with a pre-built `strategy_spec` bypasses the LLM parser (parse returns early), which is how deterministic single-asset modes like `--sweep` drive it.
- **Multi-asset portfolio** — *not* a LangGraph; [workflows/portfolio_research.py](src/trading_research_agent/workflows/portfolio_research.py) runs critique → load panel → backend → robustness → report imperatively, and the exploration/lockbox/winner-selection helpers in [workflows/explore_research.py](src/trading_research_agent/workflows/explore_research.py) are reused across both pipelines.

Layers (read these together to understand a change):

- **schemas/** — Pydantic specs whose validators enforce MVP constraints. `StrategySpec` (single-asset; families: sma_crossover, donchian_breakout, rsi_mean_reversion), `PortfolioSpec` (families: cross_sectional_momentum, dual_momentum, equal_weight_trend, time_series_momentum, volatility_scaled_momentum, crisis_hedge), `CombinedBookSpec`, plus `BacktestResult`/`BacktestMetrics`, `StrategyCritique`, `ResearchReport`.
- **nodes/** — graph nodes. LLM-backed: `parse_strategy`, `generate_slate`, `generate_portfolio_slate`, `refine_strategy`, `suggest_history_directions`, `interpret_robustness`. Everything else (`critique_strategy`, `run_backtest`, `robustness_checks`, `generate_report`) is deterministic.
- **backtesting/backends/** — `vectorbt_backend` (single-asset `Portfolio.from_signals`) and `portfolio_vectorbt` (multi-asset `Portfolio.from_orders`, cash-shared + grouped, driven by a target-weight matrix). Both add walk-forward + Monte Carlo robustness.
- **tools/** — `data_loader` (multi-source routing, below), `portfolio_signals` (pure target-weight construction), `stats` (PSR / deflated Sharpe, López de Prado), `history` (JSONL), `trial_budget`, `dates`, `indicators`, `metrics`, `plotting`.
- **reports/** — `markdown_report` (**the verdict logic** lives in `_verdict`), `html_dashboard` (read-only).
- **workflows/** — one module per CLI mode: `explore_research`, `campaign_research`, `combined_book`, `robustness_stress`, `paper_trading`, `parameter_sweep`, `macro_regime`, `portfolio_batch`, `history_detail`, `data_health`. `iterative_research` is **deprecated** (the iterate-until-pass anti-pattern).

### Data routing

`data_loader.load_ohlcv_for_asset` auto-routes a single asset by symbol: **BTC → Coin Metrics**, **Nasdaq/QQQ → FRED**, everything else → **Tiingo** (when `TIINGO_API_KEY` is set) falling back to **yfinance → Stooq**. `load_portfolio_panel` loads N assets and inner-joins them on common trading days (dropping any-NaN rows). `load_fred_series` pulls raw macro series (WALCL/FEDFUNDS/M2SL) for `--macro-regime`. Never assume a new data source works without verifying it live — a previously-removed FRED gold series silently 404'd in the past.

### Design philosophy (treat as invariants)

These are why the code is shaped the way it is. Preserve them:

- **The LLM is contained to hypothesis generation and interpretation, never the verdict.** Parsing intent and proposing strategy slates are LLM jobs; critique, backtest, robustness scoring, and the verdict are deterministic Python.
- **Out-of-sample is the gate.** Slates are *pre-registered* (committed before results), then the winner must reconfirm on a held-out lockbox (`split_date_range`, trailing fraction). In `_verdict`, **walk-forward stability is a gating check, not one vote among many** — failing it caps the verdict at `needs_more_testing`.
- **Multiple-testing is accounted for at two levels:** deflated Sharpe within a slate, and a **cumulative cross-run trial budget** ([tools/trial_budget.py](src/trading_research_agent/tools/trial_budget.py), `--budget`) that re-deflates against every portfolio shot ever logged to `outputs/history.jsonl`.
- **No look-ahead.** Signal/weight matrices are decided on bar *i* and `.shift(1)`-ed to execute on *i+1*; macro series are lagged by publication delay. Pure weight/signal functions (`portfolio_signals`, `parameter_sweep`, `macro_regime`) are unit-tested *without* vectorbt precisely so this logic is verifiable in isolation; the vectorbt execution is tested separately.
- **Strategy creation stays in the friction-ful CLI; the dashboard is read-only** (no tweak-and-rerun controls), to avoid the point-and-click overfitting ergonomic.

### Persisted state

`outputs/` (gitignored) holds `history.jsonl` (every backtested hypothesis, used by `--history`, `--budget`, `--stress`/`--paper-trade` reconstruction, and the dashboard), `paper_positions.jsonl`, equity-curve PNGs, and `dashboard.html`. History records are appended via `record_from_state`; portfolio runs use `mode="portfolio"` so reconstruction helpers (`latest_confirmed_portfolio_winner`) can find them.
