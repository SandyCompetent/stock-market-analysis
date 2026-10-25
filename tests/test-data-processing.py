"""Unit tests for data acquisition and technical indicator calculation."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.update_sample_data import WEEKDAY_TICKER_ROTATION, resolve_symbols
from src.data.fetcher import (
    amend_sample_files,
    fetch_alphavantage_news,
    fetch_live_news_data,
    fetch_newsapi_articles,
    load_sample_data,
)
from src.data.processor import (
    calculate_technical_indicators,
    create_enhanced_dataset,
    prepare_time_series_sequences,
)
from src.sentiment.finbert import FinBERTSentimentAnalyzer, aggregate_daily_sentiment
from src.utils.plotting import plot_sentiment_overview


def test_load_sample_data():
    stock_df, news_df = load_sample_data()
    assert not stock_df.empty
    assert not news_df.empty
    assert "Close" in stock_df.columns
    assert "Date" in news_df.columns or isinstance(news_df.index, pd.DatetimeIndex)


def test_calculate_technical_indicators():
    stock_df, _ = load_sample_data()
    tech_df = calculate_technical_indicators(stock_df)

    expected_cols = [
        "SMA_50",
        "SMA_200",
        "EMA_20",
        "RSI",
        "MACD_line",
        "MACD_signal",
        "BB_high",
        "BB_low",
        "BB_width",
        "OBV",
        "Pct_Change",
    ]
    for col in expected_cols:
        assert col in tech_df.columns, f"Missing indicator: {col}"

    # RSI should be bounded between 0 and 100
    assert (tech_df["RSI"] >= 0).all() and (tech_df["RSI"] <= 100).all()


def test_create_enhanced_dataset():
    stock_df, news_df = load_sample_data()
    tech_df = calculate_technical_indicators(stock_df)
    daily_sentiment_df = aggregate_daily_sentiment(news_df)

    enhanced = create_enhanced_dataset(tech_df, daily_sentiment_df)
    assert "Avg_Sentiment" in enhanced.columns
    assert "Sentiment_Ratio" in enhanced.columns
    assert "News_Count" in enhanced.columns
    assert not enhanced.isnull().all().any()


def test_prepare_time_series_sequences():
    stock_df, _ = load_sample_data()
    tech_df = calculate_technical_indicators(stock_df)
    features = ["Close", "SMA_50", "RSI", "Volume"]

    seq_len = 20
    test_size = 0.2
    X_train, X_test, y_train, y_test, scaler, test_dates = (
        prepare_time_series_sequences(
            tech_df, features, "Close", sequence_length=seq_len, test_size=test_size
        )
    )

    assert X_train.ndim == 3
    assert X_train.shape[1] == seq_len
    assert X_train.shape[2] == len(features)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(test_dates) == len(y_test)


def test_live_news_no_fallback_when_empty(tmp_path, monkeypatch):
    """Verifies that live news for AAPL does NOT fall back to NVDA sample news when keys/cache are absent."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY_1", raising=False)

    cache_dir = tmp_path / "raw_news"
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)

    # Create an NVDA sample news file in sample_dir to prove AAPL does not fall back to it
    pd.DataFrame(
        {
            "Date": ["2025-01-06"],
            "stock": ["NVDA"],
            "title": ["Nvidia launches new AI GPU"],
            "sentiment_score": [0.5],
            "sentiment_label": ["Bullish"],
        }
    ).to_csv(sample_dir / "sample-nvda-news.csv", index=False)

    df = fetch_live_news_data(
        symbol="AAPL",
        cache_dir=cache_dir,
        sample_dir=sample_dir,
    )
    assert df.empty
    assert list(df.columns) == [
        "Date",
        "stock",
        "title",
        "sentiment_score",
        "sentiment_label",
    ]


def test_live_news_incremental_cache_and_deduplication(tmp_path, monkeypatch):
    """Verifies AlphaVantage + NewsAPI fetching, FinBERT scoring, deduplication, and zero-API-call cache hits."""
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test_av_key")
    monkeypatch.setenv("NEWS_API_KEY", "test_na_key")

    cache_dir = tmp_path / "raw_news"
    sample_dir = tmp_path / "sample"
    cache_dir.mkdir(parents=True)
    sample_dir.mkdir(parents=True)

    # Seed existing cache for AAPL with an older date
    existing_cache = cache_dir / "aapl-news.csv"
    pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "stock": ["AAPL"],
            "title": ["Apple begins new year trading"],
            "sentiment_score": [0.1],
            "sentiment_label": ["Neutral"],
        }
    ).to_csv(existing_cache, index=False)

    today_str = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    today_av_str = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y%m%dT120000")

    mock_av_response = MagicMock()
    mock_av_response.status_code = 200
    mock_av_response.json.return_value = {
        "feed": [
            {
                "title": "Apple surges on record iPhone services revenue",
                "time_published": today_av_str,
                "overall_sentiment_score": 0.25,
                "overall_sentiment_label": "Somewhat-Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "ticker_sentiment_score": "0.42",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
            }
        ]
    }

    mock_na_response = MagicMock()
    mock_na_response.status_code = 200
    mock_na_response.content = b"{}"
    mock_na_response.json.return_value = {
        "status": "ok",
        "articles": [
            {
                "title": "Apple surges on record iPhone services revenue",  # duplicate title on same day
                "publishedAt": f"{today_str}T14:00:00Z",
            },
            {
                "title": "Apple announces new MacBook line with M-series chip",
                "publishedAt": f"{today_str}T16:00:00Z",
            },
        ],
    }

    analyzer = FinBERTSentimentAnalyzer(lazy_load=True)
    monkeypatch.setattr(
        analyzer, "analyze_batch", analyzer._fallback_analyze
    )

    def side_effect_get(url, params=None, timeout=30):
        if "alphavantage" in url:
            return mock_av_response
        return mock_na_response

    with patch("src.data.fetcher.requests.get", side_effect=side_effect_get) as mock_get:
        merged = fetch_live_news_data(
            symbol="AAPL",
            cache_dir=cache_dir,
            sample_dir=sample_dir,
            analyzer=analyzer,
        )
        assert mock_get.call_count == 2
        # Should have 1 existing + 2 unique new articles (duplicate deduplicated) = 3 total
        assert len(merged) == 3
        assert existing_cache.exists()

    # Subsequent call on the same day should hit local cache and make 0 API requests
    with patch("src.data.fetcher.requests.get") as mock_get_cached:
        cached_df = fetch_live_news_data(
            symbol="AAPL",
            cache_dir=cache_dir,
            sample_dir=sample_dir,
            analyzer=analyzer,
        )
        assert mock_get_cached.call_count == 0
        assert len(cached_df) == 3


def test_amend_sample_files_appends_not_replaces(tmp_path, monkeypatch):
    """Verifies amend_sample_files preserves old rows and appends new stock and news rows."""
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)

    # Seed old stock and news rows
    stock_file = sample_dir / "sample-nvda-stock.csv"
    news_file = sample_dir / "sample-nvda-news.csv"

    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [48.0, 49.0],
            "High": [49.5, 50.0],
            "Low": [47.5, 48.5],
            "Close": [49.0, 49.8],
            "Volume": [100000, 120000],
            "SP500_Close": [4700.0, 4710.0],
        }
    ).to_csv(stock_file, index=False)

    pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "stock": ["NVDA"],
            "title": ["Initial NVDA headline"],
            "sentiment_score": [0.3],
            "sentiment_label": ["Bullish"],
        }
    ).to_csv(news_file, index=False)

    mock_new_stock = pd.DataFrame(
        {
            "Open": [49.0, 51.0],
            "High": [50.0, 52.0],
            "Low": [48.5, 50.5],
            "Close": [49.8, 51.5],
            "Volume": [120000, 150000],
            "SP500_Close": [4710.0, 4750.0],
        },
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    mock_new_stock.index.name = "Date"

    with patch("src.data.fetcher.fetch_yfinance_data", return_value=mock_new_stock):
        updated_stock, updated_news = amend_sample_files(
            symbol="NVDA", sample_dir=sample_dir
        )

    assert len(updated_stock) == 3  # 2024-01-02 preserved, 2024-01-03 deduplicated, 2024-01-04 appended
    assert len(updated_news) == 1  # Initial headline preserved even when API keys are absent


def test_weekday_ticker_rotation_and_empty_plot():
    """Verifies Mon-Sat rotation mapping and empty sentiment plot resilience."""
    assert [WEEKDAY_TICKER_ROTATION[i] for i in range(6)] == [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
    ]
    assert resolve_symbols("ALL") == ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    assert resolve_symbols("aapl") == ["AAPL"]

    fig = plot_sentiment_overview(pd.DataFrame(), symbol="AAPL")
    assert fig is not None
    plt.close(fig)
