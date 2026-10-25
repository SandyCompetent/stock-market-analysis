"""Automated updater script for amending sample stock and news datasets.

Supports daily weekday rotation to stay within free-tier API quotas:
  - Monday (0):    NVDA
  - Tuesday (1):   AAPL
  - Wednesday (2): MSFT
  - Thursday (3):  GOOGL
  - Friday (4):    AMZN
  - Saturday (5):  TSLA
  - Sunday (6):    NVDA (fallback if triggered manually with AUTO on Sunday)
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fetcher import amend_sample_files

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("SampleDataUpdater")

WEEKDAY_TICKER_ROTATION = {
    0: "NVDA",   # Monday
    1: "AAPL",   # Tuesday
    2: "MSFT",   # Wednesday
    3: "GOOGL",  # Thursday
    4: "AMZN",   # Friday
    5: "TSLA",   # Saturday
    6: "NVDA",   # Sunday fallback
}


def resolve_symbols(symbol_arg: str) -> List[str]:
    """Resolves CLI --symbol input ('AUTO', 'ALL', or specific ticker) to a list of tickers."""
    cleaned = symbol_arg.strip().upper()
    if cleaned in ("", "AUTO"):
        weekday = datetime.datetime.now(datetime.timezone.utc).weekday()
        chosen = WEEKDAY_TICKER_ROTATION.get(weekday, "NVDA")
        logger.info(
            f"AUTO mode selected ticker '{chosen}' for UTC weekday {weekday}."
        )
        return [chosen]
    if cleaned == "ALL":
        return ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    # Support comma-separated tickers or single ticker
    return [s.strip().upper() for s in cleaned.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amend sample stock and news CSVs with latest market and sentiment data."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="AUTO",
        help="Ticker symbol to update, 'AUTO' for Mon-Sat rotation, or 'ALL' for all 6 tickers.",
    )
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbol)
    for sym in symbols:
        logger.info(f"=== Amending sample stock & news files for {sym} ===")
        stock_df, news_df = amend_sample_files(symbol=sym)
        logger.info(
            f"Completed {sym}: {len(stock_df)} stock rows, {len(news_df)} news rows."
        )


if __name__ == "__main__":
    main()
