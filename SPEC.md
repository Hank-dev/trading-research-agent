# Trading Algorithm Research Agent

This project converts natural-language trading ideas into structured,
reproducible backtests with an explicit anti-overfitting discipline
(pre-registered slates, held-out lockboxes, deflated Sharpe, robustness
stress-testing, and forward paper-trading).

See [README.md](README.md) for usage and [docs/WORKFLOW.md](docs/WORKFLOW.md)
for the agent pipeline. The LLM parses ideas and writes prose; it never computes
a metric or decides a verdict — all numbers come from deterministic code.
