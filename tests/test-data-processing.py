"""Unit tests for data acquisition and technical indicator calculation."""

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import load_sample_data
from src.data.processor import (
    calculate_technical_indicators,
    create_enhanced_dataset,
    prepare_time_series_sequences,
)
from src.sentiment.finbert import aggregate_daily_sentiment


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
