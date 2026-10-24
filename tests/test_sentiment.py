"""Unit tests for FinBERT sentiment scoring and daily aggregation."""

import pandas as pd
import pytest

from src.sentiment.finbert import FinBERTSentimentAnalyzer, aggregate_daily_sentiment


def test_sentiment_analyzer_fallback():
    analyzer = FinBERTSentimentAnalyzer(lazy_load=True)
    headlines = [
        "Nvidia surges to all-time record highs on massive AI profit surge",
        "Company stock plunges as revenues crash and debt misses targets",
        "Market remains flat during holiday trading session",
    ]
    sentiments, confs, scores = analyzer._fallback_analyze(headlines)

    assert len(sentiments) == 3
    assert sentiments[0] == "positive"
    assert sentiments[1] == "negative"
    assert sentiments[2] == "neutral"

    assert scores[0] > 0
    assert scores[1] < 0
    assert scores[2] == 0.0


def test_aggregate_daily_sentiment():
    mock_data = pd.DataFrame(
        {
            "Date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "title": ["Great earnings beat", "Bullish rally continues", "Stock slumps today"],
            "sentiment_score": [0.8, 0.6, -0.5],
            "sentiment_label": ["positive", "positive", "negative"],
        }
    )
    daily = aggregate_daily_sentiment(mock_data)

    assert len(daily) == 2
    assert "Avg_Sentiment" in daily.columns
    assert "Sentiment_Ratio" in daily.columns
    assert "News_Count" in daily.columns

    # Day 1 had 2 positive articles
    day1 = daily.loc[pd.to_datetime("2023-01-01")]
    assert day1["News_Count"] == 2
    assert day1["Positive_Count"] == 2
    assert day1["Negative_Count"] == 0
