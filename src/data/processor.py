"""Technical indicator computation, sentiment integration, and sequence preparation."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes technical analysis indicators on OHLCV stock data.

    Indicators:
        - SMA_50, SMA_200: Simple Moving Averages
        - EMA_20: Exponential Moving Average
        - RSI: 14-period Relative Strength Index
        - MACD_line, MACD_signal, MACD_hist: Moving Average Convergence Divergence
        - BB_high, BB_low, BB_width: Bollinger Bands
        - OBV: On-Balance Volume
        - Pct_Change: Daily Percentage Price Change (%)

    Returns:
        DataFrame with added indicators and NaN rows dropped.
    """
    data = df.copy()
    close = data["Close"]

    # 1. Daily Percentage Return
    data["Pct_Change"] = close.pct_change() * 100.0

    # 2. Moving Averages
    data["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    data["SMA_200"] = close.rolling(window=200, min_periods=1).mean()
    data["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # 3. RSI (14 periods)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))

    # 4. MACD (12-period EMA - 26-period EMA)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["MACD_line"] = ema_12 - ema_26
    data["MACD_signal"] = data["MACD_line"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD_line"] - data["MACD_signal"]

    # 5. Bollinger Bands (20 periods, 2 standard deviations)
    rolling_mean_20 = close.rolling(window=20, min_periods=1).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    data["BB_high"] = rolling_mean_20 + (2 * rolling_std_20)
    data["BB_low"] = rolling_mean_20 - (2 * rolling_std_20)
    data["BB_width"] = (data["BB_high"] - data["BB_low"]) / (rolling_mean_20 + 1e-9)

    # 6. On-Balance Volume (OBV)
    direction = np.where(
        close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0)
    )
    data["OBV"] = (direction * data["Volume"]).fillna(0).cumsum()
    
    # 7. Exogenous Variable: SP500_Close
    # If the feature is missing (e.g. sample data), backfill with Close to avoid NaNs
    if "SP500_Close" not in data.columns:
        data["SP500_Close"] = close

    # Fill any initial rolling edge cases
    data.bfill(inplace=True)
    data.ffill(inplace=True)
    return data


def create_enhanced_dataset(
    stock_df: pd.DataFrame,
    daily_sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merges technical indicators with daily sentiment scores.

    If daily_sentiment_df is missing or has missing trading days,
    neutral sentiment defaults (Avg_Sentiment=0, Sentiment_Ratio=0, News_Count=0)
    are filled smoothly.
    """
    enhanced = stock_df.copy()

    if daily_sentiment_df is None or daily_sentiment_df.empty:
        enhanced["Avg_Sentiment"] = 0.0
        enhanced["Sentiment_Ratio"] = 0.0
        enhanced["News_Count"] = 0
        return enhanced

    # Align dates
    sentiment_copy = daily_sentiment_df.copy()
    if not isinstance(sentiment_copy.index, pd.DatetimeIndex):
        if "Date" in sentiment_copy.columns:
            sentiment_copy["Date"] = pd.to_datetime(sentiment_copy["Date"])
            sentiment_copy.set_index("Date", inplace=True)

    sentiment_copy.index = pd.to_datetime(sentiment_copy.index)
    enhanced.index = pd.to_datetime(enhanced.index)

    merged = enhanced.join(sentiment_copy, how="left")

    # Fill missing sentiment attributes
    sentiment_cols = [
        "Avg_Sentiment",
        "Sentiment_Ratio",
        "News_Count",
        "Positive_Count",
        "Negative_Count",
        "Neutral_Count",
    ]
    if "Avg_Sentiment" not in merged.columns and "sentiment_score" in merged.columns:
        merged["Avg_Sentiment"] = merged["sentiment_score"]

    for col in sentiment_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0

    return merged


def prepare_time_series_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int = 60,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.Index]:
    """Generates train/test sliding window sequences while preventing lookahead leakage.

    Fits the feature scaler strictly on the training set to mirror production
    real-time inference constraints.

    Returns:
        X_train (N_train, seq_len, num_features)
        X_test  (N_test, seq_len, num_features)
        y_train (N_train,)
        y_test  (N_test,)
        target_scaler (MinMaxScaler fitted on target column)
        test_dates (pd.Index corresponding to y_test)
    """
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features missing from dataframe: {missing_cols}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Sort chronologically
    df_sorted = df.sort_index().copy()

    total_samples = len(df_sorted)
    if total_samples <= sequence_length + 10:
        raise ValueError(
            f"Not enough data rows ({total_samples}) for sequence_length ({sequence_length})."
        )

    # Chronological train/test split index
    split_idx = int(total_samples * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[
        split_idx - sequence_length :
    ]  # include lookback for first test point

    # Fit scalers strictly on training data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_df[feature_cols])

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(train_df[[target_col]])

    target_idx = feature_cols.index(target_col)

    # Transform features
    train_scaled = feature_scaler.transform(train_df[feature_cols])
    test_scaled = feature_scaler.transform(test_df[feature_cols])

    # Build sliding windows
    def _create_sequences(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            xs.append(data[i : i + sequence_length])
            ys.append(data[i + sequence_length, target_idx])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_train, y_train = _create_sequences(train_scaled)
    X_test, y_test = _create_sequences(test_scaled)

    test_dates = df_sorted.index[split_idx:]

    # Match length of dates with test labels
    test_dates = test_dates[: len(y_test)]

    logger.info(
        f"Prepared sequences: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"features={len(feature_cols)}, target='{target_col}'"
    )

    return X_train, X_test, y_train, y_test, target_scaler, test_dates
