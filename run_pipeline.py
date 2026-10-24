"""Command-line execution runner for stock data processing, sentiment analysis, and model benchmarking."""

import argparse
import logging
from pathlib import Path

from src.config import PipelineConfig, TargetMode
from src.data.fetcher import fetch_yfinance_data, load_sample_data
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def run(
    symbol: str = "NVDA",
    target_mode: str = "price",
    use_sample: bool = True,
    epochs: int = 25,
    sequence_length: int = 45,
    output_dir: str = "output",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mode = TargetMode.PRICE if target_mode.lower() == "price" else TargetMode.PERCENTAGE
    cfg = PipelineConfig(symbol=symbol, target_mode=mode)
    target_col = cfg.get_target_column()

    logger.info(f"=== Starting Stock Market AI Pipeline for {symbol} (Target: {target_col}) ===")

    # 1. Load Data
    if use_sample:
        logger.info("Loading bundled sample datasets...")
        stock_df, news_df = load_sample_data(symbol="NVDA")
    else:
        logger.info(f"Fetching live yfinance stock data for {symbol}...")
        stock_df = fetch_yfinance_data(symbol, period="2y", interval="1d")
        _, news_df = load_sample_data(symbol="NVDA")

    # 2. Process Technical Indicators
    logger.info("Computing technical indicators (SMA, EMA, RSI, MACD, BB, OBV)...")
    tech_df = calculate_technical_indicators(stock_df)

    # 3. Process News Sentiment
    logger.info("Aggregating daily FinBERT news sentiment...")
    daily_sentiment_df = aggregate_daily_sentiment(news_df)
    enhanced_df = create_enhanced_dataset(tech_df, daily_sentiment_df)

    # 4. Prepare Sequences
    baseline_feats = cfg.get_features(enhanced=False)
    enhanced_feats = cfg.get_features(enhanced=True)

    logger.info("Building time-series sequences...")
    X_tr_b, X_te_b, y_tr_b, y_te_b, scaler_b, test_dates = prepare_time_series_sequences(
        tech_df, baseline_feats, target_col, sequence_length=sequence_length, test_size=0.2
    )

    X_tr_e, X_te_e, y_tr_e, y_te_e, scaler_e, _ = prepare_time_series_sequences(
        enhanced_df, enhanced_feats, target_col, sequence_length=sequence_length, test_size=0.2
    )

    y_test_original = scaler_b.inverse_transform(y_te_b.reshape(-1, 1)).flatten()

    predictions = {}

    # 5. Train Models
    # A. Baseline LSTM
    logger.info("Training Baseline LSTM (Technical Features Only)...")
    lstm_b = LSTMTrainer(input_dim=len(baseline_feats), hidden_dim=48, num_layers=2)
    lstm_b.fit(X_tr_b, y_tr_b, epochs=epochs, batch_size=32, verbose=False)
    preds_b = scaler_b.inverse_transform(lstm_b.predict(X_te_b).reshape(-1, 1)).flatten()
    predictions["Baseline LSTM"] = preds_b

    # B. Enhanced LSTM
    logger.info("Training Enhanced LSTM (Technical + FinBERT Sentiment)...")
    lstm_e = LSTMTrainer(input_dim=len(enhanced_feats), hidden_dim=48, num_layers=2)
    lstm_e.fit(X_tr_e, y_tr_e, epochs=epochs, batch_size=32, verbose=False)
    preds_e = scaler_e.inverse_transform(lstm_e.predict(X_te_e).reshape(-1, 1)).flatten()
    predictions["Enhanced LSTM (FinBERT)"] = preds_e

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

    print("\n" + "=" * 65)
    print("               🏆 MODEL PERFORMANCE LEADERBOARD")
    print("=" * 65)
    print(leaderboard.to_string())
    print("=" * 65 + "\n")

    # 7. Save Comparison Plot
    plot_file = out_path / f"{symbol}_{target_mode}_forecast_comparison.png"
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

    leaderboard_csv = out_path / f"{symbol}_{target_mode}_leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv)
    logger.info(f"Leaderboard metrics saved to: {leaderboard_csv}")
    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stock Forecasting Pipeline")
    parser.add_argument("--symbol", type=str, default="NVDA", help="Asset ticker symbol")
    parser.add_argument(
        "--target",
        type=str,
        default="price",
        choices=["price", "percentage"],
        help="Target mode ('price' or 'percentage')",
    )
    parser.add_argument(
        "--live", action="store_true", help="Fetch live data via yfinance instead of sample"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--seq-len", type=int, default=45, help="Sequence lookback window")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory")

    args = parser.parse_args()
    run(
        symbol=args.symbol,
        target_mode=args.target,
        use_sample=not args.live,
        epochs=args.epochs,
        sequence_length=args.seq_len,
        output_dir=args.out_dir,
    )
