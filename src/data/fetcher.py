"""Data fetching and merging utility for yfinance and AlphaVantage APIs."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_yfinance_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetches historical OHLCV stock data using yfinance.

    Args:
        symbol: Stock ticker symbol (e.g. 'NVDA').
        start_date: Start date string 'YYYY-MM-DD'.
        end_date: End date string 'YYYY-MM-DD'.
        period: Alternative period if dates omitted (e.g., '1y', '2y', '5y').
        interval: Data frequency ('1d', '1wk', '1h', etc.).

    Returns:
        DataFrame indexed by Date with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    logger.info(
        f"Fetching data for {symbol} (period={period}, {start_date} to {end_date})..."
    )
    ticker = yf.Ticker(symbol)
    if start_date and end_date:
        df = ticker.history(start=start_date, end=end_date, interval=interval)
    else:
        # Use ^GSPC for S&P 500 when just GSPC is requested
        lookup_symbol = "^GSPC" if symbol.upper() == "GSPC" else symbol
        ticker = yf.Ticker(lookup_symbol)
        df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No stock data found for ticker '{symbol}'.")

    # Clean index and column casing
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


def load_sample_data(
    sample_dir: Optional[Path] = None,
    symbol: str = "NVDA",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads bundled sample stock and news datasets for zero-dependency demos.

    Returns:
        Tuple of (stock_df, news_df).
    """
    if sample_dir is None:
        sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

    stock_file = sample_dir / f"sample-{symbol.lower()}-stock.csv"
    news_file = sample_dir / f"sample-{symbol.lower()}-news.csv"

    if not stock_file.exists():
        # Fallback to any sample stock file in the folder
        stock_files = list(sample_dir.glob("sample-*-stock.csv"))
        if not stock_files:
            raise FileNotFoundError(f"Sample stock file not found in {sample_dir}")
        stock_file = stock_files[0]

    if not news_file.exists():
        news_files = list(sample_dir.glob("sample-*-news.csv"))
        if not news_files:
            raise FileNotFoundError(f"Sample news file not found in {sample_dir}")
        news_file = news_files[0]

    stock_df = pd.read_csv(stock_file)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df.set_index("Date", inplace=True)

    news_df = pd.read_csv(news_file)
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    return stock_df, news_df


def get_env_api_keys() -> List[str]:
    """Extracts sequentially defined API keys (API_KEY_1, API_KEY_2...) from environment."""
    keys = []
    i = 1
    while True:
        key = os.getenv(f"API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            break
    # Also check generic ALPHAVANTAGE_API_KEY
    generic = os.getenv("ALPHAVANTAGE_API_KEY")
    if generic and generic not in keys:
        keys.append(generic)
    return keys


def fetch_alphavantage_intraday_month(
    symbol: str,
    month: str,
    api_key: str,
    output_dir: Path,
    interval: str = "1min",
) -> bool:
    """Downloads one month of full intraday data from AlphaVantage."""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month,
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "csv",
    }
    try:
        res = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=30
        )
        res.raise_for_status()
        if "Error Message" in res.text or "Information" in res.text:
            logger.warning(f"AlphaVantage API response error: {res.text[:200]}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{month}.csv"
        out_file.write_text(res.text)
        return True
    except Exception as e:
        logger.error(f"AlphaVantage request failed: {e}")
        return False


def merge_and_sort_csvs(
    data_folder: Union[str, Path],
    output_file: Union[str, Path],
    sort_column: str = "Date",
) -> Optional[pd.DataFrame]:
    """Merges all CSV files in a directory, standardizes columns, sorts, and saves output."""
    data_folder = Path(data_folder)
    csv_files = sorted(data_folder.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {data_folder}")
        return None

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)

    # Rename common AlphaVantage timestamp columns
    rename_map = {
        "timestamp": "Date",
        "time": "Date",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    merged.rename(columns=rename_map, inplace=True)

    if sort_column in merged.columns:
        merged[sort_column] = pd.to_datetime(merged[sort_column])
        merged.sort_values(by=sort_column, ascending=True, inplace=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info(
        f"Successfully merged {len(csv_files)} files into {output_path} ({len(merged)} rows)"
    )
    return merged
