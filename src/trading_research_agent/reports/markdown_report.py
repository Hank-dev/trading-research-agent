from datetime import UTC, datetime
from pathlib import Path
import re

from pydantic import BaseModel

from trading_research_agent.config import get_output_dir
from trading_research_agent.schemas.backtest import BacktestResult
from trading_research_agent.schemas.critique import StrategyCritique
from trading_research_agent.schemas.report import ResearchReport
from trading_research_agent.schemas.strategy import StrategySpec


REQUIRED_DISCLAIMER = (
    "This is a historical research result, not a live trading recommendation."
)
BANNED_PHRASES = [
    "guaranteed profit",
    "risk-free",
    "sure win",
    "free money",
    "profitable strategy",
    "will make money",
]


def build_research_report(
    user_request: str | None,
    strategy_spec: StrategySpec | None,
    critique: StrategyCritique | None,
    backtest_result: BacktestResult | None,
    errors: list[str] | None = None,
) -> ResearchReport:
    verdict, reasons, next_tests = _verdict(backtest_result, critique, errors or [])
    markdown = _render_markdown(
        user_request=user_request,
        strategy_spec=strategy_spec,
        critique=critique,
        backtest_result=backtest_result,
        verdict=verdict,
        reasons=reasons,
        next_tests=next_tests,
        errors=errors or [],
    )
    return ResearchReport(
        markdown=_sanitize_banned_phrases(markdown),
        verdict=verdict,
        reasons=reasons,
        next_tests=next_tests,
    )


def save_markdown_report(report: ResearchReport, strategy_name: str = "research") -> ResearchReport:
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", strategy_name).strip("_").lower()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_{safe_name}_report.md"
    path.write_text(report.markdown, encoding="utf-8")
    report.report_path = str(path)
    return report


def _verdict(
    result: BacktestResult | None,
    critique: StrategyCritique | None,
    errors: list[str],
) -> tuple[str, list[str], list[str]]:
    if errors:
        return (
            "error",
            [
                "The pipeline encountered infrastructure or data errors before "
                "producing valid research evidence."
            ],
            ["Fix the reported errors and rerun the research pipeline."],
        )
    if critique is None or not critique.approved:
        return (
            "reject",
            ["The strategy did not pass deterministic critique checks."],
            ["Revise the strategy specification and rerun the critique."],
        )
    if result is None:
        return (
            "reject",
            ["Backtest was not run."],
            ["Run the deterministic backtest before evaluating the idea."],
        )

    metrics = result.metrics
    if metrics.total_return_pct < 0:
        return (
            "reject",
            ["Total return was negative."],
            ["Test whether the result persists across assets and date ranges."],
        )
    if metrics.max_drawdown_pct <= -50:
        return (
            "reject",
            ["Max drawdown was worse than or equal to -50%."],
            ["Investigate risk controls and out-of-sample behavior."],
        )
    if metrics.num_trades < 10:
        return (
            "reject",
            ["Trade count was less than 10, leaving too little evidence."],
            ["Extend the test period or evaluate additional assets."],
        )

    passed_checks = sum(check.passed for check in result.robustness_results)
    most_checks_pass = passed_checks > len(result.robustness_results) / 2
    walk_forward_failed = _walk_forward_failed(result.robustness_results)
    headline_strong = (
        metrics.total_return_pct > 0
        and metrics.beats_benchmark
        and metrics.num_trades >= 20
        and metrics.max_drawdown_pct > -50
        and most_checks_pass
    )

    # Walk-forward stability is a GATING check, not one vote among many. An edge
    # that does not hold up across sub-periods is not worth paper trading even if
    # the headline metrics look good. A walk-forward that could not run for lack
    # of data is treated the same way: we will not bless stability we never saw.
    if headline_strong and walk_forward_failed:
        return (
            "needs_more_testing",
            [
                "Headline metrics passed but walk-forward stability did not: the edge "
                "is not consistent across sub-periods (or could not be verified)."
            ],
            [
                "Lengthen the backtest so walk-forward windows are meaningful.",
                "Re-test on a held-out lockbox segment.",
                "Check whether the edge concentrates in one regime.",
            ],
        )

    if headline_strong:
        return (
            "worth_paper_trading",
            ["Positive deterministic results passed robustness checks, including walk-forward stability."],
            [
                "Test the same rules across related assets.",
                "Stress test fees and slippage.",
                "Confirm on a held-out lockbox if not already done.",
            ],
        )

    return (
        "needs_more_testing",
        ["Results were not strong enough across all MVP checks."],
        [
            "Run out-of-sample tests.",
            "Check parameter sensitivity.",
            "Compare against additional benchmarks.",
        ],
    )


def _walk_forward_failed(robustness_results: list) -> bool:
    return any(
        not check.passed and "walk-forward" in check.test_name.lower()
        for check in robustness_results
    )


def _render_markdown(
    user_request: str | None,
    strategy_spec: StrategySpec | None,
    critique: StrategyCritique | None,
    backtest_result: BacktestResult | None,
    verdict: str,
    reasons: list[str],
    next_tests: list[str],
    errors: list[str],
) -> str:
    spec_text = _model_block(strategy_spec) if strategy_spec else "No valid strategy spec."
    critique_text = _model_block(critique) if critique else "No critique was produced."

    if backtest_result:
        metrics = backtest_result.metrics
        settings = [
            f"- Asset: {backtest_result.asset}",
            f"- Period: {backtest_result.start_date} to {backtest_result.end_date}",
            f"- Engine: {backtest_result.engine}",
            "- Slippage approximation: commission plus slippage is used as effective commission.",
        ]
        results = [
            f"- Total return: {metrics.total_return_pct:.2f}%",
            f"- Buy-and-hold return: {metrics.buy_and_hold_return_pct:.2f}%",
            f"- Sharpe ratio: {_format_optional(metrics.sharpe_ratio)}",
            f"- Max drawdown: {metrics.max_drawdown_pct:.2f}%",
            f"- Number of trades: {metrics.num_trades}",
            f"- Win rate: {_format_optional(metrics.win_rate_pct, suffix='%')}",
            f"- Exposure time: {_format_optional(metrics.exposure_time_pct, suffix='%')}",
            f"- Final equity: {metrics.final_equity:.2f}",
            f"- Beats benchmark: {metrics.beats_benchmark}",
        ]
        robustness = [
            f"- {check.test_name}: {'pass' if check.passed else 'fail'} - {check.details}"
            for check in backtest_result.robustness_results
        ] or ["No robustness checks were run."]
        equity_path = backtest_result.equity_curve_path or "No equity curve was saved."
    else:
        settings = ["Backtest was not run."]
        results = ["No deterministic backtest results are available."]
        robustness = ["No robustness checks were run."]
        equity_path = "No equity curve was saved."

    hypothesis = strategy_spec.hypothesis if strategy_spec else "No hypothesis available."
    benchmark = (
        "Strategy was compared with buy-and-hold."
        if backtest_result
        else "Benchmark comparison unavailable because no backtest ran."
    )

    sections = [
        "# Strategy Research Report",
        "## User Request",
        user_request or "No user request provided.",
        "## Parsed Strategy Specification",
        spec_text,
        "## Backtesting Engine",
        "The MVP uses backtesting.py through a backend interface.",
        "## Hypothesis",
        hypothesis,
        "## Critique",
        critique_text,
        "## Backtest Settings",
        "\n".join(settings),
        "## Backtest Results",
        "\n".join(results),
        f"Equity curve: {equity_path}",
        "## Benchmark Comparison",
        benchmark,
        "## Robustness Checks",
        "\n".join(robustness),
        "## Verdict",
        verdict,
        "## Reasons",
        _bullets(reasons),
        "## Next Tests",
        _bullets(next_tests),
        "## Limitations",
        REQUIRED_DISCLAIMER,
        "The result depends on historical data quality, fixed assumptions, fees, slippage, and the tested date range.",
    ]
    if errors:
        sections.extend(["## Errors", _bullets(errors)])
    return "\n\n".join(sections) + "\n"


def _model_block(model: BaseModel) -> str:
    return "```json\n" + model.model_dump_json(indent=2) + "\n```"


def _bullets(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _format_optional(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f}{suffix}"


def _sanitize_banned_phrases(markdown: str) -> str:
    sanitized = markdown
    for phrase in BANNED_PHRASES:
        sanitized = re.sub(
            re.escape(phrase),
            "[removed unsafe claim]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized
