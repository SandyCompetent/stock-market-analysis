"""Visualization utilities for financial time series, indicators, sentiment, and forecasts."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(
    dates: pd.Index,
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    symbol: str = "NVDA",
    target_label: str = "Price ($)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots actual vs. predicted values for all models on one canvas."""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        dates,
        actual,
        label="Actual Ground Truth",
        color="black",
        linewidth=2.5,
        zorder=5,
    )

    palette = {
        "Baseline LSTM": "#1f77b4",
        "Multi-Layer LSTM": "#2ca02c",
        "Enhanced LSTM (FinBERT)": "#d62728",
        "Enhanced GRU": "#9467bd",
        "Support Vector Regressor (SVM)": "#ff7f0e",
        "ARIMA": "#8c564b",
        "Naive Persistence": "#7f7f7f",
    }

    for name, preds in predictions.items():
        color = palette.get(name, None)
        ax.plot(dates, preds, label=name, color=color, alpha=0.85, linewidth=1.8)

    ax.set_title(
        f"{symbol} - Multi-Model Forecasting Comparison", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(target_label, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_technical_summary(
    df: pd.DataFrame,
    symbol: str = "NVDA",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Renders a 3-panel dashboard of Close with Bollinger Bands, RSI, and MACD."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    # 1. Price, Moving Averages & Bollinger Bands
    ax1.plot(df.index, df["Close"], label="Close Price", color="black", linewidth=1.5)
    if "SMA_50" in df.columns:
        ax1.plot(df.index, df["SMA_50"], label="SMA 50", color="blue", alpha=0.7)
    if "BB_high" in df.columns and "BB_low" in df.columns:
        ax1.plot(
            df.index,
            df["BB_high"],
            label="Upper BB",
            color="grey",
            linestyle=":",
            alpha=0.7,
        )
        ax1.plot(
            df.index,
            df["BB_low"],
            label="Lower BB",
            color="grey",
            linestyle=":",
            alpha=0.7,
        )
        ax1.fill_between(
            df.index, df["BB_low"], df["BB_high"], color="grey", alpha=0.15
        )
    ax1.set_title(
        f"{symbol} Technical Analysis Overview", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. RSI
    if "RSI" in df.columns:
        ax2.plot(df.index, df["RSI"], color="purple", label="RSI (14)")
        ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="upper left")

    # 3. MACD
    if "MACD_line" in df.columns and "MACD_signal" in df.columns:
        ax3.plot(df.index, df["MACD_line"], label="MACD", color="blue")
        ax3.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
        if "MACD_hist" in df.columns:
            colors = ["green" if v >= 0 else "red" for v in df["MACD_hist"]]
            ax3.bar(
                df.index, df["MACD_hist"], color=colors, alpha=0.5, label="Histogram"
            )
        ax3.set_ylabel("MACD")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_sentiment_overview(
    daily_sentiment_df: pd.DataFrame,
    symbol: str = "NVDA",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots sentiment trends and news headline volume over time."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )

    # Polarity trend
    score_col = (
        "Avg_Sentiment"
        if "Avg_Sentiment" in daily_sentiment_df.columns
        else "sentiment_score"
    )
    ax1.plot(
        daily_sentiment_df.index,
        daily_sentiment_df[score_col],
        color="#2b5c8f",
        linewidth=1.5,
        label="Avg Sentiment",
    )
    ax1.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax1.set_title(
        f"{symbol} FinBERT Daily News Sentiment & Volume",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Sentiment Polarity (-1 to +1)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Volume bars
    vol_col = (
        "News_Count"
        if "News_Count" in daily_sentiment_df.columns
        else daily_sentiment_df.columns[0]
    )
    ax2.bar(
        daily_sentiment_df.index,
        daily_sentiment_df[vol_col],
        color="#5c768d",
        alpha=0.7,
        label="Headlines Count",
    )
    ax2.set_ylabel("Articles Count")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_training_loss(
    history: Dict[str, List[float]], title: str = "Model Training History"
) -> plt.Figure:
    """Plots training and validation loss curves over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss", color="blue")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(history["val_loss"], label="Validation Loss", color="orange")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig
