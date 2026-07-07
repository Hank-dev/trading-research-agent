#!/usr/bin/env python3
"""
Multi-Agent Trading Research Team
=================================

Architecture:
  Kraken (Strategy Architect)  →  generates hypotheses
  DeepSeek (Backtest Engine)   →  runs rigorous backtests (walk-forward + lockbox)
  Kraken (Evaluator)           →  reviews results, sets new research direction
  ↻ Loop for N rounds

IMPORTANT: This is RESEARCH iteration (explore new directions from failures),
NOT parameter tuning (which would be overfitting). Each round generates
QUALITATIVELY DIFFERENT strategies, not tweaks of the same one.

Usage:
  python scripts/trading_team.py --topic "momentum strategies on commodity FX" --rounds 3
  python scripts/trading_team.py --topic "volatility selling on equity indices" --rounds 2 --strategies 5

Output:
  outputs/trading_team/<timestamp>/
    round_1_hypotheses.json
    round_1_results.json
    round_1_evaluation.json
    round_2_hypotheses.json
    ...
    final_report.md
    final_report.html
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

KRAKEN_MODEL = os.getenv("KRAKEN_MODEL", "grok-4.3")
KRAKEN_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
KRAKEN_API_KEY = os.getenv("XAI_API_KEY", "")

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "trading_team"


# ═══════════════════════════════════════════════════════════════
#  KRAKEN — Strategy Architect
# ═══════════════════════════════════════════════════════════════

KRAKEN_SYSTEM = """You are Kraken, a senior quantitative strategy architect.

Your job: generate diverse, well-reasoned trading strategy hypotheses.

Rules:
1. Each hypothesis must be STRUCTURALLY DIFFERENT (different signal, different asset class, different timeframe) — not parameter variations of the same idea.
2. Base each hypothesis on a specific market inefficiency or behavioral bias, not just "it worked before."
3. Output STRICT JSON. No markdown, no commentary outside the JSON.

Output format:
{
  "strategies": [
    {
      "name": "short descriptive name",
      "hypothesis": "the market inefficiency or edge this exploits",
      "type": "single_asset | portfolio",
      "spec": {
        "family": "sma_crossover | donchian_breakout | rsi_mean_reversion | cross_sectional_momentum | dual_momentum | equal_weight_trend | time_series_momentum | volatility_scaled_momentum | crisis_hedge | cross_sectional_reversal | fx_carry",
        "assets": ["ticker1", "ticker2"],
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "parameters": { "key": "value" }
      },
      "rationale": "why this specific configuration, what regime it suits",
      "expected_behavior": "what the equity curve should look like if the edge is real"
    }
  ]
}

Asset hints: BTC=x/crypto, QQQ/SPY=US equities, TLT=bonds, GLD=gold, UUP=dollar, FXI=China, EWJ=Japan, UNG=natgas, DBA=agriculture. Use 2+ tickers for portfolio families. Date range should cover multiple regimes (bull, bear, sideways) — use at least 5 years.
"""

KRAKEN_EVAL_SYSTEM = """You are Kraken, evaluating backtest results from your research team.

Your job: review what worked, what failed, and WHY. Then set the research direction for the next round.

Critical rules:
1. Do NOT suggest tweaking parameters of failed strategies. That is overfitting.
2. If a strategy type failed, explore a COMPLETELY DIFFERENT angle.
3. If a strategy showed promise but failed robustness, note the specific failure mode and avoid that structural weakness.
4. Consider regime dependence — did it fail because the edge isn't real, or because the test period didn't favor it?
5. Weight lockbox results above all else. If it failed lockbox, the edge is likely spurious.

Output STRICT JSON:
{
  "analysis": {
    "what_worked": ["strategy names that showed promise"],
    "what_failed": ["strategy names that failed"],
    "key_insights": ["specific, actionable insights about market structure"],
    "failure_modes": ["why specific strategies failed — be precise"]
  },
  "next_round_direction": {
    "explore": ["NEW directions to try — qualitatively different from previous rounds"],
    "avoid": ["approaches that are exhausted or proven not to work"],
    "rationale": "why these new directions might contain an edge"
  },
  "should_continue": true | false,
  "stop_reason": "why to stop (if should_continue=false): e.g. 'explored all angles, no persistent edge found' or 'found robust strategy, no need to iterate'"
}
"""


def kraken_client():
    """Create the Kraken (strategy architect) LLM client."""
    return OpenAI(api_key=KRAKEN_API_KEY, base_url=KRAKEN_BASE_URL)


def kraken_generate(topic: str, round_num: int, previous_results: list | None = None) -> dict:
    """Kraken generates strategy hypotheses for this round."""
    client = kraken_client()

    if round_num == 1:
        prompt = f"""Research topic: {topic}

Generate 3-5 structurally diverse strategy hypotheses to test. Each must exploit a DIFFERENT market inefficiency."""
    else:
        prev_summary = json.dumps(previous_results[-1]["evaluation"], indent=2) if previous_results else ""
        prompt = f"""Research topic: {topic}
Round: {round_num}

Previous round results:
{prev_summary}

Generate 3-5 NEW strategy hypotheses based on the evaluation above. These must be QUALITATIVELY DIFFERENT from previous rounds. Do not repeat approaches that failed."""

    response = client.chat.completions.create(
        model=KRAKEN_MODEL,
        messages=[
            {"role": "system", "content": KRAKEN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # higher temp for diverse hypotheses
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def kraken_evaluate(round_num: int, hypotheses: dict, results: list) -> dict:
    """Kraken reviews backtest results and decides next direction."""
    client = kraken_client()

    # Summarize results for Kraken (strip verbose data, keep signal)
    result_summary = []
    for r in results:
        entry = {
            "name": r.get("name", "unknown"),
            "verdict": r.get("verdict", "unknown"),
            "metrics": r.get("metrics", {}),
            "lockbox": r.get("lockbox_verdict", "not tested"),
            "robustness_failures": r.get("robustness_failures", []),
            "walk_forward_stable": r.get("walk_forward_stable", None),
        }
        result_summary.append(entry)

    prompt = f"""Round {round_num} results:

Hypotheses tested:
{json.dumps(hypotheses.get("strategies", []), indent=2)[:2000]}

Backtest results:
{json.dumps(result_summary, indent=2)[:3000]}

Evaluate these results and set direction for the next round."""

    response = client.chat.completions.create(
        model=KRAKEN_MODEL,
        messages=[
            {"role": "system", "content": KRAKEN_EVAL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # lower temp for analytical evaluation
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    return json.loads(content)


# ═══════════════════════════════════════════════════════════════
#  DEEPSEEK — Backtest Engine
# ═══════════════════════════════════════════════════════════════

def deepseek_backtest(strategy: dict) -> dict:
    """
    Run a single strategy through the trading-research-agent backtest engine.
    Uses the existing infrastructure: vectorbt + walk-forward + lockbox + robustness.
    """
    spec = strategy.get("spec", {})
    name = strategy.get("name", "unnamed")
    stype = strategy.get("type", "single_asset")

    try:
        if stype == "portfolio":
            return _run_portfolio_backtest(strategy)
        else:
            return _run_single_asset_backtest(strategy)
    except Exception as e:
        return {
            "name": name,
            "verdict": "error",
            "error": str(e),
            "metrics": {},
        }


def _run_single_asset_backtest(strategy: dict) -> dict:
    """Run via the single-asset research graph."""
    from trading_research_agent.workflows.research_graph import build_research_graph
    from trading_research_agent.schemas.strategy import StrategySpec

    spec = strategy["spec"]
    name = strategy["name"]

    # Build a natural-language description for the parser
    family = spec.get("family", "sma_crossover")
    assets = spec.get("assets", ["BTC"])
    start = spec.get("start_date", "2018-01-01")
    end = spec.get("end_date", "2025-12-31")
    params = spec.get("parameters", {})

    # Use the CLI approach — run the actual engine
    import subprocess

    cmd = [
        sys.executable, "-m", "trading_research_agent.app",
        "--save-report",
        f"{family} strategy on {assets[0] if len(assets)==1 else ','.join(assets)}",
        start, "to", end,
    ]

    # Add flags based on available modes
    cmd_str = f'python -m trading_research_agent.app "{family} on {",".join(assets)} {start} to {end}" --save-report'

    result = subprocess.run(
        cmd_str,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,  # 5 min max per strategy
        env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
    )

    # Parse output for verdict and metrics
    output = result.stdout + result.stderr
    return _parse_backtest_output(name, output, result.returncode)


def _run_portfolio_backtest(strategy: dict) -> dict:
    """Run via the portfolio research pipeline."""
    import subprocess

    spec = strategy["spec"]
    name = strategy["name"]
    family = spec.get("family", "cross_sectional_momentum")
    assets = spec.get("assets", ["SPY", "TLT", "GLD"])
    start = spec.get("start_date", "2015-01-01")
    end = spec.get("end_date", "2025-12-31")

    cmd_str = (
        f'python -m trading_research_agent.app '
        f'--portfolio --explore 1 '
        f'"{family} on {",".join(assets)} {start} to {end}"'
    )

    result = subprocess.run(
        cmd_str,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=300,
        env={**os.environ, "NUMBA_DISABLE_JIT": "1"},
    )

    output = result.stdout + result.stderr
    return _parse_backtest_output(name, output, result.returncode)


def _parse_backtest_output(name: str, output: str, returncode: int) -> dict:
    """Extract verdict and key metrics from the CLI output."""
    result = {
        "name": name,
        "verdict": "unknown",
        "metrics": {},
        "lockbox_verdict": "not tested",
        "robustness_failures": [],
        "walk_forward_stable": None,
        "raw_output": output[-2000:] if len(output) > 2000 else output,
    }

    # Parse verdict
    for line in output.split("\n"):
        line_lower = line.lower().strip()
        if "verdict:" in line_lower or "verdict" in line_lower and ":" in line:
            for v in ["confirmed", "needs_more_testing", "rejected", "failed"]:
                if v in line_lower:
                    result["verdict"] = v
                    break

    # Parse key metrics
    for line in output.split("\n"):
        line_lower = line.lower()
        if "sharpe" in line_lower and ":" in line:
            try:
                val = line.split(":")[-1].strip().split()[0]
                result["metrics"]["sharpe"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "max drawdown" in line_lower or "max_drawdown" in line_lower:
            try:
                val = line.split(":")[-1].strip().split()[0].rstrip("%")
                result["metrics"]["max_drawdown"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "cagr" in line_lower or "total return" in line_lower:
            try:
                val = line.split(":")[-1].strip().split()[0].rstrip("%")
                result["metrics"]["cagr"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "win rate" in line_lower:
            try:
                val = line.split(":")[-1].strip().split()[0].rstrip("%")
                result["metrics"]["win_rate"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "lockbox" in line_lower and ("fail" in line_lower or "pass" in line_lower):
            result["lockbox_verdict"] = "passed" if "pass" in line_lower else "failed"
        elif "walk-forward" in line_lower or "walk_forward" in line_lower:
            result["walk_forward_stable"] = "stable" not in line_lower or "pass" in line_lower

    if returncode != 0 and result["verdict"] == "unknown":
        result["verdict"] = "error"

    return result


# ═══════════════════════════════════════════════════════════════
#  DEEPSEEK ANALYST — Result Analysis (optional, if API key set)
# ═══════════════════════════════════════════════════════════════

def deepseek_analyze(results: list) -> str | None:
    """If DeepSeek API is configured, get a code-level analysis of failures."""
    if not DEEPSEEK_API_KEY:
        return None

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""Analyze these backtest results. For each strategy, explain in 2-3 sentences WHY it likely succeeded or failed. Focus on market microstructure, regime dependence, and statistical validity.

Results:
{json.dumps(results, indent=2)[:3000]}
"""
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a quantitative analyst. Be concise and precise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATION LOOP
# ═══════════════════════════════════════════════════════════════

def run_team(topic: str, rounds: int = 3, strategies_per_round: int = 4) -> dict:
    """
    Main orchestration loop.
    
    Kraken generates → DeepSeek backtests → Kraken evaluates → repeat
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = []
    all_results = []

    print(f"\n{'═'*60}")
    print(f"  TRADING RESEARCH TEAM")
    print(f"  Topic: {topic}")
    print(f"  Rounds: {rounds} | Strategies/round: {strategies_per_round}")
    print(f"  Kraken model: {KRAKEN_MODEL}")
    print(f"  DeepSeek model: {DEEPSEEK_MODEL if DEEPSEEK_API_KEY else 'deterministic engine only'}")
    print(f"{'═'*60}\n")

    for round_num in range(1, rounds + 1):
        print(f"\n{'─'*40}")
        print(f"  ROUND {round_num}/{rounds}")
        print(f"{'─'*40}")

        # ─── KRAKEN: Generate hypotheses ───
        print(f"\n🐙 Kraken generating hypotheses...")
        try:
            hypotheses = kraken_generate(topic, round_num, all_rounds if round_num > 1 else None)
        except Exception as e:
            print(f"   ✗ Kraken failed: {e}")
            break

        strategies = hypotheses.get("strategies", [])[:strategies_per_round]
        print(f"   → {len(strategies)} strategies generated:")
        for s in strategies:
            print(f"     • {s['name']} ({s.get('type','?')}): {s.get('hypothesis','')[:80]}")

        # Save hypotheses
        (run_dir / f"round_{round_num}_hypotheses.json").write_text(
            json.dumps(hypotheses, indent=2)
        )

        # ─── DEEPSEEK: Backtest each strategy ───
        print(f"\n⚡ DeepSeek running backtests...")
        round_results = []
        for i, strategy in enumerate(strategies):
            print(f"   [{i+1}/{len(strategies)}] Testing: {strategy['name']}...", end=" ", flush=True)
            result = deepseek_backtest(strategy)
            round_results.append(result)
            print(f"{result.get('verdict', '?')}")

            # Print key metrics
            metrics = result.get("metrics", {})
            if metrics:
                metric_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
                print(f"         {metric_str}")

        # Save results
        (run_dir / f"round_{round_num}_results.json").write_text(
            json.dumps(round_results, indent=2)
        )

        # ─── Optional: DeepSeek analysis ───
        analysis = deepseek_analyze(round_results)
        if analysis:
            print(f"\n🤖 DeepSeek analysis:")
            for line in analysis.split("\n"):
                print(f"   {line}")

        # ─── KRAKEN: Evaluate ───
        print(f"\n🐙 Kraken evaluating results...")
        try:
            evaluation = kraken_evaluate(round_num, hypotheses, round_results)
        except Exception as e:
            print(f"   ✗ Evaluation failed: {e}")
            evaluation = {"error": str(e), "should_continue": False}

        (run_dir / f"round_{round_num}_evaluation.json").write_text(
            json.dumps(evaluation, indent=2)
        )

        # Print evaluation
        analysis_data = evaluation.get("analysis", {})
        if analysis_data.get("key_insights"):
            print(f"\n   Key insights:")
            for insight in analysis_data.get("key_insights", []):
                print(f"     → {insight}")

        if analysis_data.get("failure_modes"):
            print(f"   Failure modes:")
            for fm in analysis_data.get("failure_modes", []):
                print(f"     ✗ {fm}")

        # Store round
        round_data = {
            "round": round_num,
            "hypotheses": hypotheses,
            "results": round_results,
            "evaluation": evaluation,
            "deepseek_analysis": analysis,
        }
        all_rounds.append(round_data)
        all_results.extend(round_results)

        # Check if Kraken says stop
        if not evaluation.get("should_continue", True):
            print(f"\n⛔ Kraken says stop: {evaluation.get('stop_reason', 'no reason given')}")
            break

    # ─── Generate final report ───
    print(f"\n{'═'*60}")
    print(f"  GENERATING FINAL REPORT")
    print(f"{'═'*60}")

    report = _generate_report(topic, all_rounds, all_results)
    (run_dir / "final_report.md").write_text(report)
    print(f"\n✓ Report saved: {run_dir / 'final_report.md'}")
    print(f"✓ All data saved: {run_dir}/")

    return {"rounds": all_rounds, "report": report, "output_dir": str(run_dir)}


def _generate_report(topic: str, rounds: list, all_results: list) -> str:
    """Generate a markdown summary report."""
    lines = [
        f"# Trading Research Report: {topic}",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Rounds: {len(rounds)}",
        f"Total strategies tested: {len(all_results)}",
        "",
    ]

    # Summary stats
    verdicts = {}
    for r in all_results:
        v = r.get("verdict", "unknown")
        verdicts[v] = verdicts.get(v, 0) + 1

    lines.append("## Verdict Summary")
    lines.append("")
    for v, count in sorted(verdicts.items()):
        lines.append(f"- **{v}**: {count}")
    lines.append("")

    # Best performers
    confirmed = [r for r in all_results if r.get("verdict") in ("confirmed", "needs_more_testing")]
    if confirmed:
        lines.append("## Promising Strategies")
        lines.append("")
        for r in confirmed:
            lines.append(f"### {r['name']}")
            lines.append(f"- Verdict: {r['verdict']}")
            metrics = r.get("metrics", {})
            if metrics:
                for k, v in metrics.items():
                    lines.append(f"- {k}: {v}")
            lines.append(f"- Lockbox: {r.get('lockbox_verdict', 'N/A')}")
            lines.append("")

    # Per-round breakdown
    for rd in rounds:
        lines.append(f"## Round {rd['round']}")
        lines.append("")

        # Strategies tested
        for s in rd["hypotheses"].get("strategies", []):
            lines.append(f"### {s['name']}")
            lines.append(f"- Hypothesis: {s.get('hypothesis', 'N/A')}")
            lines.append(f"- Rationale: {s.get('rationale', 'N/A')[:200]}")
            lines.append("")

        # Evaluation
        ev = rd.get("evaluation", {})
        analysis = ev.get("analysis", {})
        if analysis.get("key_insights"):
            lines.append("**Key insights:**")
            for insight in analysis["key_insights"]:
                lines.append(f"- {insight}")
            lines.append("")

        if analysis.get("failure_modes"):
            lines.append("**Failure modes:**")
            for fm in analysis["failure_modes"]:
                lines.append(f"- {fm}")
            lines.append("")

        direction = ev.get("next_round_direction", {})
        if direction.get("rationale"):
            lines.append(f"**Next direction:** {direction['rationale']}")
            lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    last_eval = rounds[-1].get("evaluation", {}) if rounds else {}
    if last_eval.get("stop_reason"):
        lines.append(last_eval["stop_reason"])
    elif confirmed:
        lines.append(f"Found {len(confirmed)} promising strategies out of {len(all_results)} tested. Further validation recommended with extended lockbox periods.")
    else:
        lines.append(f"No strategies survived robustness testing across {len(all_results)} candidates. The explored edge likely does not persist out-of-sample.")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by Multi-Agent Trading Research Team (Kraken + DeepSeek)*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Trading Research Team")
    parser.add_argument("topic", help="Research topic / market area to explore")
    parser.add_argument("--rounds", type=int, default=3, help="Number of research rounds (default: 3)")
    parser.add_argument("--strategies", type=int, default=4, help="Strategies per round (default: 4)")
    args = parser.parse_args()

    if not KRAKEN_API_KEY:
        print("ERROR: XAI_API_KEY not set in .env")
        sys.exit(1)

    result = run_team(args.topic, args.rounds, args.strategies)
    print(f"\nDone. Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
