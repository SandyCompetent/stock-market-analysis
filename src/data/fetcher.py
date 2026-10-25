"""Data fetching, local caching, and sample amendment utility for yfinance, AlphaVantage, and NewsAPI."""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import requests
import requests_cache
import yfinance as yf

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.sentiment.finbert import FinBERTSentimentAnalyzer

# Install requests-cache globally for yfinance
requests_cache.install_cache("yfinance_cache", expire_after=3600)

logger = logging.getLogger(__name__)

NEWS_COLUMNS = ["Date", "stock", "title", "sentiment_score", "sentiment_label"]
PLACEHOLDER_KEYS = {
    "",
    "your_alphavantage_api_key_here",
    "your_newsapi_key_here",
    "none",
    "null",
}


def _get_secret(key_name: str) -> Optional[str]:
    """Safely retrieves a secret from environment variables or Streamlit st.secrets."""
    val = os.getenv(key_name, "").strip()
    if val and val.lower() not in PLACEHOLDER_KEYS:
        return val

    if "streamlit" in sys.modules:
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                st_val = str(st.secrets.get(key_name, "")).strip()
                if st_val and st_val.lower() not in PLACEHOLDER_KEYS:
                    return st_val
        except Exception:
            pass

    return None


def get_env_api_keys() -> List[str]:
    """Extracts AlphaVantage API keys (API_KEY_1..N and ALPHAVANTAGE_API_KEY) from env or st.secrets."""
    keys: List[str] = []
    i = 1
    while True:
        key = _get_secret(f"API_KEY_{i}")
        if key:
            if key not in keys:
                keys.append(key)
            i += 1
        else:
            break

    generic = _get_secret("ALPHAVANTAGE_API_KEY")
    if generic and generic not in keys:
        keys.append(generic)
    return keys


def get_news_api_key() -> Optional[str]:
    """Extracts NEWS_API_KEY from environment or Streamlit secrets."""
    return _get_secret("NEWS_API_KEY")


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
        DataFrame indexed by Date with columns ['Open', 'High', 'Low', 'Close', 'Volume', 'SP500_Close'].
    """
    logger.info(
        f"Fetching data for {symbol} (period={period}, {start_date} to {end_date})..."
    )
    # Use ^GSPC for S&P 500 when just GSPC is requested, otherwise fetch symbol
    lookup_symbol = "^GSPC" if symbol.upper() == "GSPC" else symbol

    # We fetch the target stock and SP500 together
    if lookup_symbol != "^GSPC":
        tickers = f"{lookup_symbol} ^GSPC"
    else:
        tickers = "^GSPC"

    if start_date and end_date:
        raw_df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
        )
    else:
        raw_df = yf.download(
            tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
        )

    if raw_df.empty:
        raise ValueError(f"No stock data found for ticker '{symbol}'.")

    # If multiple tickers were requested, yf returns a MultiIndex column DataFrame
    if lookup_symbol != "^GSPC":
        df = raw_df[lookup_symbol].copy()
        sp500_close = raw_df["^GSPC"]["Close"]
    else:
        # single ticker downloaded
        df = raw_df.copy()
        sp500_close = df["Close"]

    # Clean index and column casing
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    cols = ["Open", "High", "Low", "Close", "Volume"]
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols].copy()

    # Drop rows where Close is NaN
    if "Close" in df.columns:
        df.dropna(subset=["Close"], inplace=True)

    # Add SP500 Close
    sp500_close.index = pd.to_datetime(sp500_close.index).tz_localize(None)
    df["SP500_Close"] = sp500_close
    df["SP500_Close"] = df["SP500_Close"].ffill().bfill()

    return df


def load_sample_data(
    sample_dir: Optional[Path] = None,
    symbol: str = "NVDA",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads bundled sample stock and news datasets for offline demos.

    Returns:
        Tuple of (stock_df, news_df).
    """
    if sample_dir is None:
        sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

    stock_file = sample_dir / f"sample-{symbol.lower()}-stock.csv"
    news_file = sample_dir / f"sample-{symbol.lower()}-news.csv"

    if not stock_file.exists():
        # Fallback to any sample stock file in the folder only in offline sample mode
        stock_files = sorted(sample_dir.glob("sample-*-stock.csv"))
        if not stock_files:
            raise FileNotFoundError(f"Sample stock file not found in {sample_dir}")
        stock_file = stock_files[0]

    if not news_file.exists():
        news_files = sorted(sample_dir.glob("sample-*-news.csv"))
        if not news_files:
            raise FileNotFoundError(f"Sample news file not found in {sample_dir}")
        news_file = news_files[0]

    stock_df = pd.read_csv(stock_file)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df.set_index("Date", inplace=True)

    news_df = pd.read_csv(news_file)
    if not news_df.empty and "Date" in news_df.columns:
        news_df["Date"] = pd.to_datetime(news_df["Date"])

    return stock_df, news_df


def fetch_alphavantage_news(
    symbol: str,
    api_key: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetches historical news & sentiment from AlphaVantage NEWS_SENTIMENT API.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL').
        api_key: AlphaVantage API key.
        start_date: Optional ISO start date 'YYYY-MM-DD' to fetch incrementally from.
        limit: Max articles to retrieve (up to 1000).

    Returns:
        DataFrame with columns ['Date', 'stock', 'title', 'sentiment_score', 'sentiment_label'].
    """
    symbol_upper = symbol.upper()
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol_upper,
        "limit": str(limit),
        "sort": "LATEST",
        "apikey": api_key,
    }
    if start_date:
        dt = pd.to_datetime(start_date)
        params["time_from"] = dt.strftime("%Y%m%dT0000")

    try:
        res = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=30
        )
        res.raise_for_status()
        payload = res.json()
    except Exception as e:
        logger.warning(f"AlphaVantage NEWS_SENTIMENT request failed for {symbol_upper}: {e}")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    if not isinstance(payload, dict):
        return pd.DataFrame(columns=NEWS_COLUMNS)

    if "Information" in payload or "Error Message" in payload or "Note" in payload:
        msg = (
            payload.get("Information")
            or payload.get("Error Message")
            or payload.get("Note")
        )
        logger.warning(f"AlphaVantage API notice for {symbol_upper}: {msg}")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    feed = payload.get("feed", [])
    if not feed:
        logger.info(f"AlphaVantage returned 0 news articles for {symbol_upper}.")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    rows = []
    for item in feed:
        title = str(item.get("title", "")).strip()
        time_pub = str(item.get("time_published", "")).strip()
        if not title or not time_pub:
            continue

        # Parse YYYYMMDDTHHMMSS date part
        pub_date = pd.to_datetime(time_pub[:8], format="%Y%m%d", errors="coerce")
        if pd.isna(pub_date):
            pub_date = pd.to_datetime(time_pub, errors="coerce")
        if pd.isna(pub_date):
            continue

        # Prefer ticker-specific sentiment for the target symbol when present
        score = item.get("overall_sentiment_score", 0.0)
        label = item.get("overall_sentiment_label", "Neutral")
        for ts in item.get("ticker_sentiment", []):
            if str(ts.get("ticker", "")).upper() == symbol_upper:
                score = ts.get("ticker_sentiment_score", score)
                label = ts.get("ticker_sentiment_label", label)
                break

        try:
            score_float = float(score)
        except (TypeError, ValueError):
            score_float = 0.0

        rows.append(
            {
                "Date": pub_date.normalize(),
                "stock": symbol_upper,
                "title": title,
                "sentiment_score": round(score_float, 6),
                "sentiment_label": str(label),
            }
        )

    logger.info(f"Fetched {len(rows)} articles from AlphaVantage for {symbol_upper}.")
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def fetch_newsapi_articles(
    symbol: str,
    api_key: str,
    start_date: Optional[str] = None,
    page_size: int = 100,
    analyzer: Optional[FinBERTSentimentAnalyzer] = None,
) -> pd.DataFrame:
    """Fetches recent news articles from NewsAPI (/v2/everything) and scores them with FinBERT.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL').
        api_key: NewsAPI key.
        start_date: Optional ISO start date 'YYYY-MM-DD' (clamped to 28 days for free tier compatibility).
        page_size: Number of articles to request (max 100).
        analyzer: Optional FinBERTSentimentAnalyzer instance for scoring headlines.

    Returns:
        DataFrame with columns ['Date', 'stock', 'title', 'sentiment_score', 'sentiment_label'].
    """
    symbol_upper = symbol.upper()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    min_free_tier_date = today - pd.Timedelta(days=28)

    if start_date:
        requested_dt = pd.to_datetime(start_date).tz_localize(None).normalize()
        effective_start = max(requested_dt, min_free_tier_date)
    else:
        effective_start = min_free_tier_date

    params = {
        "q": f'"{symbol_upper}" stock OR "{symbol_upper}"',
        "from": effective_start.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(page_size, 100),
        "apiKey": api_key,
    }

    try:
        res = requests.get(
            "https://newsapi.org/v2/everything", params=params, timeout=30
        )
        payload = res.json() if res.content else {}
        if res.status_code != 200 or payload.get("status") != "ok":
            msg = payload.get("message", f"HTTP {res.status_code}")
            logger.warning(f"NewsAPI response warning for {symbol_upper}: {msg}")
            return pd.DataFrame(columns=NEWS_COLUMNS)
    except Exception as e:
        logger.warning(f"NewsAPI request failed for {symbol_upper}: {e}")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    articles = payload.get("articles", [])
    if not articles:
        logger.info(f"NewsAPI returned 0 articles for {symbol_upper}.")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    valid_dates = []
    valid_titles = []
    for art in articles:
        title = str(art.get("title") or "").strip()
        pub_at = str(art.get("publishedAt") or "").strip()
        if not title or title == "[Removed]" or not pub_at:
            continue

        pub_date = pd.to_datetime(pub_at[:10], format="%Y-%m-%d", errors="coerce")
        if pd.isna(pub_date):
            continue

        valid_dates.append(pub_date.normalize())
        valid_titles.append(title)

    if not valid_titles:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    sentiment_engine = analyzer or FinBERTSentimentAnalyzer(lazy_load=True)
    labels, _, scores = sentiment_engine.analyze_batch(valid_titles)

    rows = [
        {
            "Date": dt,
            "stock": symbol_upper,
            "title": ttl,
            "sentiment_score": round(float(sc), 6),
            "sentiment_label": str(lbl).capitalize(),
        }
        for dt, ttl, sc, lbl in zip(valid_dates, valid_titles, scores, labels)
    ]

    logger.info(
        f"Fetched and FinBERT-scored {len(rows)} articles from NewsAPI for {symbol_upper}."
    )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _merge_and_deduplicate_news(dfs: List[pd.DataFrame], symbol: str) -> pd.DataFrame:
    """Combines multiple news DataFrames, normalizes dates, deduplicates by (Date, title), and sorts."""
    non_empty = [d for d in dfs if d is not None and not d.empty]
    if not non_empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    combined = pd.concat(non_empty, ignore_index=True)

    if "Date" not in combined.columns:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.tz_localize(
        None
    ).dt.normalize()
    combined["title"] = combined["title"].astype(str).str.strip()
    combined = combined.dropna(subset=["Date", "title"])
    combined = combined[combined["title"] != ""]

    if "stock" not in combined.columns:
        combined["stock"] = symbol.upper()
    else:
        combined["stock"] = combined["stock"].fillna(symbol.upper())

    if "sentiment_score" not in combined.columns:
        combined["sentiment_score"] = 0.0
    else:
        combined["sentiment_score"] = pd.to_numeric(
            combined["sentiment_score"], errors="coerce"
        ).fillna(0.0)

    if "sentiment_label" not in combined.columns:
        combined["sentiment_label"] = "Neutral"
    else:
        combined["sentiment_label"] = combined["sentiment_label"].fillna("Neutral")

    combined = combined[NEWS_COLUMNS]
    combined.drop_duplicates(subset=["Date", "title"], keep="first", inplace=True)
    combined.sort_values(by="Date", ascending=True, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _save_news_csv(df: pd.DataFrame, path: Path) -> None:
    """Writes a news DataFrame to disk with YYYY-MM-DD formatted Date strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if not out.empty and "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def fetch_live_news_data(
    symbol: str,
    cache_dir: Optional[Path] = None,
    sample_dir: Optional[Path] = None,
    force_refresh: bool = False,
    analyzer: Optional[FinBERTSentimentAnalyzer] = None,
) -> pd.DataFrame:
    """Fetches and caches live news for `symbol` from AlphaVantage and NewsAPI without sample fallback.

    1. Checks `data/raw/news/{symbol}-news.csv` and exact-match `data/sample/sample-{symbol}-news.csv`.
       Never falls back to another ticker's sample file.
    2. If cached data already includes today's date (and `force_refresh=False`), returns cached data
       with 0 external API calls.
    3. Otherwise queries AlphaVantage and NewsAPI only for dates since the last cached date,
       merges, deduplicates, saves to `data/raw/news/{symbol}-news.csv`, and returns the result.
    4. If no cached data exists and APIs yield no data, returns an empty DataFrame (no fallback).
    """
    symbol_upper = symbol.upper()
    project_root = Path(__file__).resolve().parent.parent.parent

    if cache_dir is None:
        cache_dir = project_root / "data" / "raw" / "news"
    if sample_dir is None:
        sample_dir = project_root / "data" / "sample"

    cache_file = cache_dir / f"{symbol_upper.lower()}-news.csv"
    exact_sample_file = sample_dir / f"sample-{symbol_upper.lower()}-news.csv"

    seed_dfs: List[pd.DataFrame] = []
    if exact_sample_file.exists():
        try:
            seed_dfs.append(pd.read_csv(exact_sample_file))
        except Exception as e:
            logger.warning(f"Failed to read exact sample news file {exact_sample_file}: {e}")

    if cache_file.exists():
        try:
            seed_dfs.append(pd.read_csv(cache_file))
        except Exception as e:
            logger.warning(f"Failed to read cached news file {cache_file}: {e}")

    existing_df = _merge_and_deduplicate_news(seed_dfs, symbol=symbol_upper)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    last_date: Optional[pd.Timestamp] = None
    if not existing_df.empty:
        last_date = pd.to_datetime(existing_df["Date"].max()).normalize()
        if not force_refresh and last_date >= today:
            logger.info(
                f"Using up-to-date local news cache for {symbol_upper} "
                f"({len(existing_df)} articles through {last_date.strftime('%Y-%m-%d')})."
            )
            if not cache_file.exists():
                _save_news_csv(existing_df, cache_file)
            return existing_df

    start_date_str = (
        last_date.strftime("%Y-%m-%d")
        if last_date is not None
        else (today - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    )

    fetched_dfs: List[pd.DataFrame] = [existing_df]

    # 1. Query AlphaVantage NEWS_SENTIMENT
    av_keys = get_env_api_keys()
    if av_keys:
        for av_key in av_keys:
            av_df = fetch_alphavantage_news(
                symbol=symbol_upper,
                api_key=av_key,
                start_date=start_date_str,
            )
            if not av_df.empty:
                fetched_dfs.append(av_df)
                break
    else:
        logger.info(
            f"ALPHAVANTAGE_API_KEY not configured; skipping AlphaVantage news fetch for {symbol_upper}."
        )

    # 2. Query NewsAPI /v2/everything
    news_api_key = get_news_api_key()
    if news_api_key:
        na_df = fetch_newsapi_articles(
            symbol=symbol_upper,
            api_key=news_api_key,
            start_date=start_date_str,
            analyzer=analyzer,
        )
        if not na_df.empty:
            fetched_dfs.append(na_df)
    else:
        logger.info(
            f"NEWS_API_KEY not configured; skipping NewsAPI fetch for {symbol_upper}."
        )

    merged_df = _merge_and_deduplicate_news(fetched_dfs, symbol=symbol_upper)

    if not merged_df.empty:
        _save_news_csv(merged_df, cache_file)
        logger.info(
            f"Saved {len(merged_df)} total news records for {symbol_upper} to {cache_file}."
        )
    else:
        logger.warning(
            f"No live or cached news data available for {symbol_upper}. Proceeding without news data (no fallback)."
        )

    return merged_df


def amend_sample_files(
    symbol: str = "NVDA",
    sample_dir: Optional[Path] = None,
    analyzer: Optional[FinBERTSentimentAnalyzer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Incrementally amends (appends & deduplicates, never replaces) both stock and news sample CSVs.

    Updates:
        - `data/sample/sample-{symbol}-stock.csv`
        - `data/sample/sample-{symbol}-news.csv`

    Returns:
        Tuple of (updated_stock_df, updated_news_df).
    """
    symbol_upper = symbol.upper()
    if sample_dir is None:
        sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    stock_file = sample_dir / f"sample-{symbol_upper.lower()}-stock.csv"
    news_file = sample_dir / f"sample-{symbol_upper.lower()}-news.csv"
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    # --- 1. Amend Stock CSV ---
    existing_stock = pd.DataFrame()
    start_stock_date: Optional[str] = None
    if stock_file.exists():
        try:
            existing_stock = pd.read_csv(stock_file)
            if not existing_stock.empty and "Date" in existing_stock.columns:
                existing_stock["Date"] = pd.to_datetime(
                    existing_stock["Date"]
                ).dt.tz_localize(None).dt.normalize()
                last_dt = existing_stock["Date"].max()
                if pd.notna(last_dt):
                    start_stock_date = last_dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Could not read existing stock file {stock_file}: {e}")

    try:
        if start_stock_date:
            end_stock_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_stock = fetch_yfinance_data(
                symbol_upper,
                start_date=start_stock_date,
                end_date=end_stock_date,
                interval="1d",
            ).reset_index()
        else:
            new_stock = fetch_yfinance_data(
                symbol_upper, period="2y", interval="1d"
            ).reset_index()
    except Exception as e:
        logger.warning(f"yfinance fetch during sample amend for {symbol_upper}: {e}")
        new_stock = pd.DataFrame()

    if not new_stock.empty:
        new_stock["Date"] = pd.to_datetime(new_stock["Date"]).dt.tz_localize(
            None
        ).dt.normalize()

    if not existing_stock.empty and not new_stock.empty:
        # Keep consistent columns; preserve standard OHLCV (+ SP500_Close if present)
        stock_combined = pd.concat([existing_stock, new_stock], ignore_index=True)
    elif not new_stock.empty:
        stock_combined = new_stock.copy()
    else:
        stock_combined = existing_stock.copy()

    if not stock_combined.empty:
        stock_combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
        stock_combined.sort_values(by="Date", ascending=True, inplace=True)
        if "SP500_Close" in stock_combined.columns:
            stock_combined["SP500_Close"] = (
                stock_combined["SP500_Close"].ffill().bfill()
            )
        stock_to_save = stock_combined.copy()
        stock_to_save["Date"] = pd.to_datetime(stock_to_save["Date"]).dt.strftime(
            "%Y-%m-%d"
        )
        stock_to_save.to_csv(stock_file, index=False)
        logger.info(
            f"Amended stock sample file {stock_file} ({len(stock_combined)} total rows)."
        )
        stock_combined.set_index("Date", inplace=True)

    # --- 2. Amend News CSV ---
    updated_news = fetch_live_news_data(
        symbol=symbol_upper,
        cache_dir=sample_dir,
        sample_dir=sample_dir,
        force_refresh=True,
        analyzer=analyzer,
    )
    # Write directly to sample-{symbol}-news.csv (and remove any redundant raw filename in sample_dir)
    raw_named_in_sample = sample_dir / f"{symbol_upper.lower()}-news.csv"
    if raw_named_in_sample.exists() and raw_named_in_sample != news_file:
        raw_named_in_sample.unlink()

    if not updated_news.empty or not news_file.exists():
        _save_news_csv(updated_news, news_file)
        logger.info(
            f"Amended news sample file {news_file} ({len(updated_news)} total rows)."
        )

    return stock_combined, updated_news


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
