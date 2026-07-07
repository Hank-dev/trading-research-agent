# Multi-Agent Trading Research Team

## For Hummer (OpenClaw)

Dette systemet lar deg kjøre et autonomt trading research team med to agent-roller:

### Arkitektur

```
Topic → Kraken (Grok) → genererer 3-5 strategihypotheser
                              ↓
        DeepSeek (backtest engine) → tester hver strategi
                              ↓
        Kraken (evaluator) → vurderer resultater, setter ny retning
                              ↓
                    ↻ Loop N runder
                              ↓
                    Final report (markdown)
```

### Roller

**Kraken (Strategy Architect)**
- Modell: Grok via xAI API
- Rolle: Genererer strategihypotheser, vurderer resultater
- Output: Structured JSON med strategy specs
- Hver runde: QUALITATIVT FORSKJELLIGE strategier (ikke parameter-tuning)

**DeepSeek (Backtest Engine)**
- Modell: DeepSeek API (hvis konfigurert) eller deterministisk engine
- Rolle: Kjører backtests via trading-research-agent infrastrukturen
- Inkluderer: walk-forward, lockbox, robustness checks
- Output: Verdict + metrics per strategi

### Viktig: Anti-Overfit Design

Systemet er eksplisitt designet for å UNNGÅ overfitting:
- Hver runde genererer nye strategier i FORSKJELLIGE retninger (ikke tweaks)
- Lockbox er ultimate gate — feiler den, er strategien død
- Kraken prompten forbyr parameter-tuning av feilede strategier
- Multiple-testing correction via trial budget (eksisterer i koden)

### Bruk

```bash
cd /home/johannes/trading-research-agent
source .venv/bin/activate

# Kjør 3 runder med 4 strategier per runde
NUMBA_DISABLE_JIT=1 python scripts/trading_team.py "momentum on commodity FX (AUD, NZD, CAD)" --rounds 3 --strategies 4

# Kjør 2 runder med 5 strategier
NUMBA_DISABLE_JIT=1 python scripts/trading_team.py "volatility selling on equity indices" --rounds 2 --strategies 5
```

### Output

```
outputs/trading_team/<timestamp>/
  round_1_hypotheses.json     ← Kraken's strategier
  round_1_results.json         ← DeepSeek's backtest results
  round_1_evaluation.json      ← Kraken's vurdering
  round_2_hypotheses.json
  ...
  final_report.md              ← Sluttrapport
```

### Miljøvariabler (.env)

Påkrevd:
- `XAI_API_KEY` — for Kraken (Grok)

Valgfritt:
- `DEEPSEEK_API_KEY` — for DeepSeek analyse (hvis ikke satt, bruker kun deterministisk engine)
- `KRAKEN_MODEL` — default: grok-4.3
- `DEEPSEEK_MODEL` — default: deepseek-chat
- `TIINGO_API_KEY` — for markedsdata (påkrevd for de fleste strategier)

### Cron Setup (autonom kjøring)

Kan settes opp som cron via Hermes for å kjøre f.eks. hver natt:

```python
cronjob(action='create',
        schedule='0 22 * * *',  # hver kveld kl 22
        prompt='Run the trading research team on a rotating topic. Check /home/johannes/trading-research-agent/scripts/trading_team.py for usage. Pick a topic from: commodity FX, equity momentum, volatility selling, cross-asset rotation. Run 2 rounds with 3 strategies each.',
        deliver='origin',
        workdir='/home/johannes/trading-research-agent')
```

### Troubleshooting

- **NUMBA_DISABLE_JIT=1** er påkrevd under Python 3.14
- Hvis Tiingo mangler, fungerer kun BTC og FRED-data (QQQ/SPY)
- Timeout per strategi: 5 minutter
- Hvis Kraken API feiler, avbrytes runden gracefully
