"""Retry caching tickers that hit Tiingo's hourly rate limit.

Checks if already cached, fetches only missing ones.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from trading_research_agent.tools.data_loader import load_ohlcv_tiingo, tiingo_cache_status

REST_TICKERS = ["EWZ", "VNQ", "IBIT", "GBTC", "TQQQ", "UPRO"]
START = "2005-01-04"
END = "2026-06-19"


def main():
    print(f"Retrying {len(REST_TICKERS)} tickers...\n")
    total = len(REST_TICKERS)
    fetched = 0
    skipped = 0
    failed = 0

    for i, ticker in enumerate(REST_TICKERS, 1):
        status = tiingo_cache_status(ticker, START, END)
        if status.get("covered"):
            rows = status.get("rows", 0)
            print(f"  [{i}/{total}] {ticker:5s} ✓ already cached ({rows} rows)")
            skipped += 1
            continue

        print(f"  [{i}/{total}] {ticker:5s} → fetching...", end="", flush=True)
        try:
            df = load_ohlcv_tiingo(ticker, START, END)
            rows = len(df)
            print(f" ✓ {rows} rows")
            fetched += 1
        except Exception as e:
            err = str(e).strip()
            print(f" ✗ FAILED: {err}")
            failed += 1

        if i < total or failed > 0:
            time.sleep(2)

    print(f"\nResult: {fetched} fetched, {skipped} already cached, {failed} failed")
    return failed


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
