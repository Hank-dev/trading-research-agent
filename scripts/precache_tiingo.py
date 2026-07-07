"""Pre-cache a broad set of common tickers from Tiingo so backtests hit local cache.

Run once: python scripts/precache_tiingo.py
Each ticker = 1 API call (500/day free tier).
"""

import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from trading_research_agent.tools.data_loader import load_ohlcv_tiingo, tiingo_cache_status

TICKERS = [
    # --- Major US Indices ---
    "SPY", "QQQ", "IWM", "DIA",
    "VTI", "VOO",
    # --- Sectors ---
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLU", "XLY", "XLP", "XLB", "XLRE",
    # --- Fixed Income ---
    "TLT", "IEF", "SHY",
    "AGG", "BND",
    "LQD", "HYG",
    "BIL", "SHV",
    "TIP", "MUB", "MBB",
    # --- Gold / Precious / Commodities ---
    "GLD", "IAU",
    "SLV",
    "GDX",
    "USO", "DBC", "DBA",
    # --- Currency ETFs ---
    "FXE", "FXY", "FXB", "FXF", "FXA", "FXC",
    "UUP",
    # --- Volatility ---
    "VIXY",
    # --- International ---
    "EEM", "VWO",
    "EFA", "VEA",
    "EWJ", "EWG", "EWW", "EWZ",
    # --- REITs ---
    "VNQ",
    # --- Crypto-adjacent ---
    "IBIT", "GBTC",
    # --- Leveraged (for reference/risk-parity variants) ---
    "TQQQ", "UPRO",
]

START = "2005-01-04"
END = "2026-06-19"


def main():
    total = len(TICKERS)
    success = 0
    skipped = 0
    failed = 0

    print(f"Pre-caching {total} tickers from {START} to {END} via Tiingo\n")

    for i, ticker in enumerate(TICKERS, 1):
        status = tiingo_cache_status(ticker, START, END)
        if status.get("covered"):
            row_count = status.get("rows", 0)
            print(f"  [{i}/{total}] {ticker:5s} ✓ already cached ({row_count} rows)")
            skipped += 1
            continue

        print(f"  [{i}/{total}] {ticker:5s} → fetching...", end="", flush=True)
        try:
            df = load_ohlcv_tiingo(ticker, START, END)
            rows = len(df)
            print(f" ✓ {rows} rows")
            success += 1
        except Exception as e:
            print(f" ✗ FAILED: {e}")
            failed += 1

        # Rate-limit courtesy: 500/day = ~1 req per 3 sec is safe
        time.sleep(2)

    print(f"\nDone: {success} fetched, {skipped} cached, {failed} failed / {total} total")


if __name__ == "__main__":
    main()
