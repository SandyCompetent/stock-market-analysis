"""Interactive Streamlit Demo for Stock Market Analysis & Sentiment-Enhanced Forecasting."""


import matplotlib.pyplot as plt
import streamlit as st

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
from src.utils.plotting import (
    plot_model_comparison,
    plot_sentiment_overview,
    plot_technical_summary,
)

# Page configuration
st.set_page_config(
    page_title="Stock Market AI & FinBERT Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📈 Stock Market AI: Multimodal Forecasting Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Integrates <b>FinBERT NLP Financial News Sentiment</b> with <b>Technical Indicators</b> across <b>LSTM, GRU, SVR, and ARIMA</b> time-series models.</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration & Controls")

ticker = st.sidebar.selectbox(
    "Select Asset Ticker", ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], index=0
)

data_mode = st.sidebar.radio(
    "Data Source",
    ["Curated Sample (Instant / Offline)", "Live Data (yfinance)"],
    index=0,
    help="Curated sample includes pre-matched FinBERT sentiment for zero-dependency instant demo.",
)

data_interval_choice = st.sidebar.selectbox(
    "Data Frequency / Interval",
    ["1 Day (1d)", "1 Hour (1h)", "1 Week (1wk)"],
    index=0,
    help="Select bar interval frequency. (Note: 1 Hour interval is restricted to max 2 Years by Yahoo Finance).",
)

interval_map = {
    "1 Day (1d)": "1d",
    "1 Hour (1h)": "1h",
    "1 Week (1wk)": "1wk",
}
selected_interval = interval_map[data_interval_choice]

# Dynamically filter historical duration options based on interval choice
if selected_interval == "1h":
    duration_options = ["1 Year", "2 Years"]
    default_duration_idx = 1
    duration_help = "For 1 Hour interval data, Yahoo Finance API limits range to 2 Years (730 days) max."
else:
    duration_options = ["1 Year", "2 Years", "3 Years", "4 Years", "5 Years"]
    default_duration_idx = 1
    duration_help = "Select historical time range used for feature calculation, model training, and testing."

history_period_choice = st.sidebar.selectbox(
    "Historical Data Duration",
    duration_options,
    index=default_duration_idx,
    help=duration_help,
)

period_map = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "4 Years": "4y",
    "5 Years": "5y",
}
selected_period = period_map[history_period_choice]

target_mode_choice = st.sidebar.radio(
    "Target Prediction Variable",
    ["Price ($ Close)", "Daily Percentage Change (%)"],
    index=0,
)
target_mode = (
    TargetMode.PRICE if "Price" in target_mode_choice else TargetMode.PERCENTAGE
)

seq_len = st.sidebar.slider(
    "Lookback Window (Sequence Length)", min_value=15, max_value=90, value=45, step=5
)
epochs = st.sidebar.slider(
    "LSTM Training Epochs", min_value=5, max_value=50, value=25, step=5
)

st.sidebar.markdown("---")
st.sidebar.subheader("Models to Evaluate")
run_baseline_lstm = st.sidebar.checkbox("Baseline LSTM (Technical Only)", value=True)
run_enhanced_lstm = st.sidebar.checkbox(
    "Enhanced LSTM (+ FinBERT Sentiment)", value=True
)
run_svm = st.sidebar.checkbox("Support Vector Regressor (SVM)", value=True)
run_arima = st.sidebar.checkbox("ARIMA Time Series", value=True)
run_naive = st.sidebar.checkbox("Naive Persistence Benchmark", value=True)

run_button = st.sidebar.button("🚀 Run Forecast & Benchmark", type="primary")


# --- Data Loading Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(
    selected_ticker: str, mode: str, period: str = "2y", interval: str = "1d"
):
    if "Sample" in mode:
        stock_df, news_df = load_sample_data(symbol=selected_ticker)
        return stock_df, news_df
    else:
        stock_df = fetch_yfinance_data(
            selected_ticker, period=period, interval=interval
        )
        news_df = fetch_live_news_data(symbol=selected_ticker)
        return stock_df, news_df


with st.spinner("Loading dataset and calculating indicators..."):
    stock_df, news_df = get_data(ticker, data_mode, selected_period, selected_interval)
    tech_df = calculate_technical_indicators(stock_df)
    daily_sentiment_df = aggregate_daily_sentiment(news_df)
    has_news_data = not news_df.empty and not daily_sentiment_df.empty
    enhanced_df = (
        create_enhanced_dataset(tech_df, daily_sentiment_df) if has_news_data else None
    )

# Top metrics cards
col1, col2, col3, col4 = st.columns(4)
latest_close = tech_df["Close"].iloc[-1]
prev_close = tech_df["Close"].iloc[-2]
pct_change_day = ((latest_close - prev_close) / prev_close) * 100
avg_sentiment_val = (
    daily_sentiment_df["Avg_Sentiment"].mean() if not daily_sentiment_df.empty else 0.0
)

col1.metric("Latest Close", f"${latest_close:.2f}", f"{pct_change_day:+.2f}%")
col2.metric("Trading Bars Analyzed", f"{len(tech_df)} bars ({selected_interval})")
col3.metric("News Headlines Processed", f"{len(news_df)} articles")
col4.metric("Avg Sentiment Score", f"{avg_sentiment_val:+.3f}", "FinBERT Polarity")

# Navigation Tabs
tab_forecast, tab_tech, tab_sentiment, tab_methodology = st.tabs(
    [
        "🏆 Model Forecasts & Benchmark",
        "📊 Technical Indicators",
        "📰 FinBERT News Sentiment",
        "🔬 Methodology & Architecture",
    ]
)

# --- Tab 1: Forecasts & Leaderboard ---
with tab_forecast:
    cfg = PipelineConfig(
        symbol=ticker,
        target_mode=target_mode,
    )
    target_col = cfg.get_target_column()
    target_label = (
        "Price ($ Close)" if target_mode == TargetMode.PRICE else "Daily Percentage Change (%)"
    )

    st.subheader(f"Model Predictions vs. Ground Truth ({target_label})")
    st.info(
        f"🎯 **Active Target Mode**: Predicting **{target_label}** for **{ticker}** | "
        f"**Historical Duration**: {selected_period} | **Interval**: {selected_interval}"
    )

    # Sequence preparation
    baseline_feats = cfg.get_features(enhanced=False)
    enhanced_feats = cfg.get_features(enhanced=True)

    X_train_b, X_test_b, y_train_b, y_test_b, scaler_b, test_dates = (
        prepare_time_series_sequences(
            tech_df, baseline_feats, target_col, sequence_length=seq_len, test_size=0.2
        )
    )

    # Actual values in original scale
    y_test_original = scaler_b.inverse_transform(y_test_b.reshape(-1, 1)).flatten()

    predictions_dict = {}

    status_container = st.empty()

    # Train / Run models
    if run_baseline_lstm:
        status_container.info("🧠 Training Baseline LSTM...")
        trainer_b = LSTMTrainer(
            input_dim=len(baseline_feats), hidden_dim=48, num_layers=2
        )
        trainer_b.fit(X_train_b, y_train_b, epochs=epochs, batch_size=32, verbose=False)
        preds_b_scaled = trainer_b.predict(X_test_b)
        predictions_dict["Baseline LSTM"] = scaler_b.inverse_transform(
            preds_b_scaled.reshape(-1, 1)
        ).flatten()

    if run_enhanced_lstm:
        if has_news_data and enhanced_df is not None:
            X_train_e, X_test_e, y_train_e, y_test_e, scaler_e, _ = (
                prepare_time_series_sequences(
                    enhanced_df,
                    enhanced_feats,
                    target_col,
                    sequence_length=seq_len,
                    test_size=0.2,
                )
            )
            status_container.info("⚡ Training Sentiment-Enhanced LSTM...")
            trainer_e = LSTMTrainer(
                input_dim=len(enhanced_feats), hidden_dim=48, num_layers=2
            )
            trainer_e.fit(
                X_train_e, y_train_e, epochs=epochs, batch_size=32, verbose=False
            )
            preds_e_scaled = trainer_e.predict(X_test_e)
            predictions_dict["Enhanced LSTM (FinBERT)"] = scaler_e.inverse_transform(
                preds_e_scaled.reshape(-1, 1)
            ).flatten()
        else:
            st.warning(
                f"⚠️ No news data available for **{ticker}** — skipping **Enhanced LSTM (FinBERT)** (no fallback used)."
            )

    if run_svm:
        status_container.info("📐 Training Support Vector Regressor (SVM)...")
        svm = SVMPredictor(C=50.0)
        svm.fit(X_train_b, y_train_b)
        svm_preds_scaled = svm.predict(X_test_b)
        predictions_dict["Support Vector Regressor (SVM)"] = scaler_b.inverse_transform(
            svm_preds_scaled.reshape(-1, 1)
        ).flatten()

    if run_arima:
        status_container.info("📈 Fitting ARIMA statistical model...")
        arima = ARIMAPredictor(order=(3, 1, 0))
        arima.fit(y_train_b)
        arima_scaled = arima.forecast(steps=len(y_test_b), last_history=y_train_b)
        predictions_dict["ARIMA"] = scaler_b.inverse_transform(
            arima_scaled.reshape(-1, 1)
        ).flatten()

    if run_naive:
        naive = NaivePredictor.predict(y_test_original)
        predictions_dict["Naive Persistence"] = naive

    status_container.empty()

    # Comparison Plot
    fig_comp = plot_model_comparison(
        dates=test_dates,
        actual=y_test_original,
        predictions=predictions_dict,
        symbol=ticker,
        target_label=target_label,
    )
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # Leaderboard Table
    st.markdown("### 🏆 Performance Leaderboard")
    is_pct = target_mode == TargetMode.PERCENTAGE
    leaderboard = compare_models(
        predictions_dict, y_test_original, is_percentage=is_pct
    )

    st.dataframe(
        leaderboard.style.highlight_min(
            subset=["RMSE", "MAE", "MAPE (%)"], color="#d4edda"
        ).highlight_max(subset=["Directional Accuracy (%)"], color="#d4edda"),
        use_container_width=True,
    )

    # Key takeaways
    best_model = leaderboard.index[0]
    best_rmse = leaderboard.loc[best_model, "RMSE"]
    best_da = leaderboard.loc[best_model, "Directional Accuracy (%)"]
    st.success(
        f"💡 **Top Performer**: **{best_model}** achieved the leading performance score with an RMSE of **{best_rmse}** and Directional Accuracy of **{best_da}%**."
    )

# --- Tab 2: Technical Indicators ---
with tab_tech:
    st.subheader(f"{ticker} Technical Indicator Analysis")
    fig_tech = plot_technical_summary(tech_df, symbol=ticker)
    st.pyplot(fig_tech)
    plt.close(fig_tech)

    with st.expander("🔍 View Indicator Data Table"):
        st.dataframe(tech_df.tail(20), use_container_width=True)

# --- Tab 3: News Sentiment ---
with tab_sentiment:
    st.subheader("FinBERT Financial Sentiment Intelligence")
    if has_news_data:
        fig_sent = plot_sentiment_overview(daily_sentiment_df, symbol=ticker)
        st.pyplot(fig_sent)
        plt.close(fig_sent)

        st.markdown("### 📰 Recent Headlines & Sentiment Classification")
        display_cols = [
            c
            for c in ["Date", "title", "sentiment_label", "sentiment_score"]
            if c in news_df.columns
        ]
        st.dataframe(news_df[display_cols].tail(50), use_container_width=True)
    else:
        st.info(
            f"ℹ️ No news headlines available for **{ticker}**. "
            "Configure `ALPHAVANTAGE_API_KEY` and/or `NEWS_API_KEY` in `.env` or Streamlit Secrets to fetch live news."
        )

# --- Tab 4: Methodology ---
with tab_methodology:
    st.markdown(
        """
        ### System Architecture & Pipeline Design

        ```
        [ Market Data (yfinance / AlphaVantage) ] ──> [ Technical Indicator Engine ] ─┐
                                                                                       ├──> [ Scaler & Sequence Builder ] ──> [ Multi-Model Forecasting ]
        [ News Headlines (AlphaVantage News) ] ────> [ FinBERT NLP Sentiment Engine ] ─┘
        ```

        #### 1. Data Layer
        - **Stock Time Series**: Historical OHLCV downloaded via `yfinance` or automated AlphaVantage scrapers.
        - **Feature Engineering**: SMA (50, 200), EMA (20), RSI (14), MACD (12, 26, 9), Bollinger Bands (20, 2σ), and On-Balance Volume (OBV).

        #### 2. FinBERT Sentiment Engine
        - Utilizes `ProsusAI/finbert`, a BERT model specifically fine-tuned on financial phrasebanks.
        - Produces 3-class classification (*Positive*, *Negative*, *Neutral*) and a continuous polarity score:
        $$\\text{Polarity} = P(\\text{Positive}) - P(\\text{Negative}) \\in [-1.0, +1.0]$$
        - Aggregated daily with headline volume and sentiment ratio metrics.

        #### 3. Forecasting Models
        - **Baseline LSTM**: 2-layer Recurrent Neural Network trained on technical indicators.
        - **Enhanced LSTM**: Recurrent Neural Network fed with combined technical + FinBERT sentiment vectors.
        - **Support Vector Regressor (SVR)**: Non-linear kernel regression on flattened historical sequences.
        - **ARIMA**: Autoregressive Integrated Moving Average statistical model.
        - **Naive Baseline**: Persistence benchmark ($y_t = y_{t-1}$).

        #### 4. Evaluation Metrics
        - **Root Mean Squared Error (RMSE)**: Penalizes large forecasting errors.
        - **Directional Accuracy**: Percentage of correct market trend predictions (Up vs Down).
        """
    )
