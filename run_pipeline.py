"""Command-line execution runner for stock data processing, sentiment analysis, and model benchmarking."""

import argparse
import logging
from pathlib import Path

from src.config import PipelineConfig, TargetMode
from src.data.fetcher import (
    fetch_live_news_data,
    fetch_yfinance_data,
    load_sample_data,
)
from src.data.processor import (
    calculate_technical_indicators,
    create_enhanced_dataset,
    prepare_time_series_sequences,
)
from src.models.baselines import ARIMAPredictor, NaivePredictor, SVMPredictor
from src.models.evaluation import compare_models
from src.models.lstm import LSTMTrainer
from src.sentiment.finbert import aggregate_daily_sentiment
from src.utils.plotting import plot_model_comparison

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("PipelineRunner")


def run(
    symbol: str = "NVDA",
    target_mode: str = "price",
    use_sample: bool = True,
    epochs: int = 25,
    sequence_length: int = 45,
    hidden_dim: int = 48,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    output_dir: str = "output",
    save_output: bool = True,
):
    out_path = Path(output_dir)
    if save_output:
        out_path.mkdir(parents=True, exist_ok=True)

    mode = TargetMode.PRICE if target_mode.lower() == "price" else TargetMode.PERCENTAGE
    cfg = PipelineConfig(symbol=symbol, target_mode=mode)
    target_col = cfg.get_target_column()

    logger.info(
        f"=== Starting Stock Market AI Pipeline for {symbol} (Target: {target_col}) ==="
    )

    # 1. Load Data
    if use_sample:
        logger.info(f"Loading bundled sample datasets for {symbol}...")
        stock_df, news_df = load_sample_data(symbol=symbol)
    else:
        logger.info(f"Fetching live yfinance stock data for {symbol}...")
        stock_df = fetch_yfinance_data(symbol, period="2y", interval="1d")
        logger.info(
            f"Fetching live/cached news data for {symbol} (AlphaVantage + NewsAPI, no fallback)..."
        )
        news_df = fetch_live_news_data(symbol=symbol)

    # 2. Process Technical Indicators
    logger.info("Computing technical indicators (SMA, EMA, RSI, MACD, BB, OBV)...")
    tech_df = calculate_technical_indicators(stock_df)

    # 3. Process News Sentiment
    has_news = news_df is not None and not news_df.empty
    if has_news:
        logger.info("Aggregating daily FinBERT news sentiment...")
        daily_sentiment_df = aggregate_daily_sentiment(news_df)
        has_news = not daily_sentiment_df.empty
    else:
        daily_sentiment_df = None

    # 4. Prepare Sequences
    baseline_feats = cfg.get_features(enhanced=False)
    enhanced_feats = cfg.get_features(enhanced=True)

    logger.info("Building time-series sequences...")
    X_tr_b, X_te_b, y_tr_b, y_te_b, scaler_b, test_dates = (
        prepare_time_series_sequences(
            tech_df,
            baseline_feats,
            target_col,
            sequence_length=sequence_length,
            test_size=0.2,
        )
    )

    y_test_original = scaler_b.inverse_transform(y_te_b.reshape(-1, 1)).flatten()

    predictions = {}

    # 5. Train Models
    # A. Baseline LSTM
    logger.info("Training Baseline LSTM (Technical Features Only)...")
    lstm_b = LSTMTrainer(
        input_dim=len(baseline_feats), hidden_dim=hidden_dim, num_layers=num_layers
    )
    lstm_b.fit(
        X_tr_b,
        y_tr_b,
        epochs=epochs,
        batch_size=32,
        learning_rate=learning_rate,
        verbose=False,
    )
    preds_b = scaler_b.inverse_transform(
        lstm_b.predict(X_te_b).reshape(-1, 1)
    ).flatten()
    predictions["Baseline LSTM"] = preds_b

    # B. Enhanced LSTM (Only if news data is available)
    if has_news:
        enhanced_df = create_enhanced_dataset(tech_df, daily_sentiment_df)
        X_tr_e, X_te_e, y_tr_e, y_te_e, scaler_e, _ = prepare_time_series_sequences(
            enhanced_df,
            enhanced_feats,
            target_col,
            sequence_length=sequence_length,
            test_size=0.2,
        )
        logger.info("Training Enhanced LSTM (Technical + FinBERT Sentiment)...")
        lstm_e = LSTMTrainer(
            input_dim=len(enhanced_feats), hidden_dim=hidden_dim, num_layers=num_layers
        )
        lstm_e.fit(
            X_tr_e,
            y_tr_e,
            epochs=epochs,
            batch_size=32,
            learning_rate=learning_rate,
            verbose=False,
        )
        preds_e = scaler_e.inverse_transform(
            lstm_e.predict(X_te_e).reshape(-1, 1)
        ).flatten()
        predictions["Enhanced LSTM (FinBERT)"] = preds_e
    else:
        logger.warning(
            f"No news data available for {symbol} — skipping Enhanced LSTM (FinBERT)."
        )

    # C. SVM
    logger.info("Fitting Support Vector Regressor (SVM)...")
    svm = SVMPredictor(C=50.0)
    svm.fit(X_tr_b, y_tr_b)
    preds_svm = scaler_b.inverse_transform(svm.predict(X_te_b).reshape(-1, 1)).flatten()
    predictions["Support Vector Regressor (SVM)"] = preds_svm

    # D. ARIMA
    logger.info("Fitting ARIMA time series model...")
    arima = ARIMAPredictor(order=(3, 1, 0))
    arima.fit(y_tr_b)
    preds_arima = scaler_b.inverse_transform(
        arima.forecast(steps=len(y_te_b), last_history=y_tr_b).reshape(-1, 1)
    ).flatten()
    predictions["ARIMA"] = preds_arima

    # E. Naive
    predictions["Naive Persistence"] = NaivePredictor.predict(y_test_original)

    # 6. Evaluation Leaderboard
    is_pct = mode == TargetMode.PERCENTAGE
    leaderboard = compare_models(predictions, y_test_original, is_percentage=is_pct)

    # Calculate dataset date ranges for the report
    split_idx = int(len(tech_df) * 0.8)
    train_dates = tech_df.index[:split_idx]
    train_start = train_dates.min().strftime("%Y-%m-%d") if not train_dates.empty else "Unknown"
    train_end = train_dates.max().strftime("%Y-%m-%d") if not train_dates.empty else "Unknown"
    test_start = test_dates.min().strftime("%Y-%m-%d") if not test_dates.empty else "Unknown"
    test_end = test_dates.max().strftime("%Y-%m-%d") if not test_dates.empty else "Unknown"

    print("\n" + "=" * 80)
    print(f"    🏆 MODEL PERFORMANCE LEADERBOARD | {symbol} | Predicting: {target_mode.capitalize()}    ")
    print(f"    📅 Train Period: {train_start} to {train_end} | Test Period (Predicted): {test_start} to {test_end}")
    print("=" * 80)
    print(leaderboard.to_string())
    print("=" * 80 + "\n")

    # 7. Save Results (if enabled)
    if save_output:
        plot_file = out_path / f"{symbol.lower()}-{target_mode}-forecast-comparison.png"
        target_lbl = "Price ($)" if mode == TargetMode.PRICE else "Percentage Change (%)"
        fig = plot_model_comparison(
            dates=test_dates,
            actual=y_test_original,
            predictions=predictions,
            symbol=symbol,
            target_label=target_lbl,
            save_path=str(plot_file),
        )
        logger.info(f"Comparison plot saved to: {plot_file}")

        leaderboard_csv = out_path / f"{symbol.lower()}-{target_mode}-leaderboard.csv"
        leaderboard.to_csv(leaderboard_csv)
        logger.info(f"Leaderboard metrics saved to: {leaderboard_csv}")
    
    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stock Forecasting Pipeline")
    parser.add_argument(
        "--symbol", type=str, default="NVDA", help="Asset ticker symbol"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        choices=["price", "percentage"],
        help="Target mode ('price' or 'percentage')",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live data via yfinance instead of sample",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--seq-len", type=int, default=45, help="Sequence lookback window"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=48, help="Hidden dimensions for LSTM"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for AdamW"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots and CSV leaderboard to output directory",
    )
    parser.add_argument(
        "--out-dir", type=str, default="output", help="Output directory"
    )

    args = parser.parse_args()
    run(
        symbol=args.symbol,
        target_mode=args.target,
        use_sample=not args.live,
        epochs=args.epochs,
        sequence_length=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        output_dir=args.out_dir,
        save_output=not args.no_save,
    )
