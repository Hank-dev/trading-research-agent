# VPS Deployment

This project is a CLI research tool, not a long-running trading bot. On a VPS,
run it as one-shot commands, and optionally schedule `--report-html` to refresh
the static dashboard.

## Option 1: Docker Compose

1. Create `.env` from `.env.example` and fill in API keys.
2. Build the image:

```bash
docker compose build
```

3. Run commands through Compose:

```bash
docker compose run --rm trade-research --history
docker compose run --rm trade-research --report-html
docker compose run --rm trade-research --portfolio-batch examples/portfolio_batch.json --lockbox-pct 0.2
```

Compose mounts `./outputs` into the container, so history, reports, charts, the
dashboard, and cache survive container rebuilds.

## Option 2: Native Python + systemd

These examples use:

- App directory: `/opt/trading-research-agent`
- Environment file: `/etc/trading-research-agent/trading-research-agent.env`
- Persistent data: `/var/lib/trading-research-agent`
- Service user: `trading-research`

Create the user and directories:

```bash
sudo useradd --system --create-home --home-dir /opt/trading-research-agent --shell /usr/sbin/nologin trading-research
sudo mkdir -p /etc/trading-research-agent /var/lib/trading-research-agent/outputs /var/lib/trading-research-agent/cache
sudo chown -R trading-research:trading-research /opt/trading-research-agent /var/lib/trading-research-agent
```

Place this repository at `/opt/trading-research-agent`, then install the app
from that directory:

```bash
cd /opt/trading-research-agent
sudo -u trading-research python3.11 -m venv .venv
sudo -u trading-research .venv/bin/pip install --upgrade pip
sudo -u trading-research .venv/bin/pip install .
```

Create the environment file from `deploy/trading-research-agent.env.example`,
fill in secrets, and lock down permissions:

```bash
sudo install -m 640 -o root -g trading-research deploy/trading-research-agent.env.example /etc/trading-research-agent/trading-research-agent.env
sudo editor /etc/trading-research-agent/trading-research-agent.env
```

Smoke test:

```bash
sudo -u trading-research sh -c 'set -a; . /etc/trading-research-agent/trading-research-agent.env; set +a; exec /opt/trading-research-agent/.venv/bin/trade-research --report-html'
```

Install the daily dashboard refresh timer:

```bash
sudo cp deploy/systemd/trade-research-dashboard.service /etc/systemd/system/
sudo cp deploy/systemd/trade-research-dashboard.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trade-research-dashboard.timer
sudo systemctl start trade-research-dashboard.service
```

Check logs:

```bash
journalctl -u trade-research-dashboard.service -n 100 --no-pager
```

## Optional: Serve The Dashboard With nginx

The generated dashboard is a static HTML file. The included nginx example serves
`/var/lib/trading-research-agent/outputs/dashboard.html` and blocks JSONL/cache
files.

```bash
sudo cp deploy/nginx/trading-research-dashboard.conf /etc/nginx/sites-available/trading-research-dashboard
sudo ln -s /etc/nginx/sites-available/trading-research-dashboard /etc/nginx/sites-enabled/trading-research-dashboard
sudo nginx -t
sudo systemctl reload nginx
```

Set a real `server_name` and HTTPS certificate before exposing it publicly.

## Persistence

Set these in the VPS environment file:

```bash
TRADING_RESEARCH_OUTPUT_DIR=/var/lib/trading-research-agent/outputs
TRADING_RESEARCH_CACHE_DIR=/var/lib/trading-research-agent/cache
```

`TRADING_RESEARCH_OUTPUT_DIR` controls history, reports, charts, paper positions,
and `dashboard.html`. `TRADING_RESEARCH_CACHE_DIR` controls market-data cache.
