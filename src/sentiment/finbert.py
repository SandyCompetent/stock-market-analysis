"""FinBERT sentiment analyzer and daily aggregation engine."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """Detects available hardware acceleration (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBERTSentimentAnalyzer:
    """FinBERT (ProsusAI/finbert) sentiment inference engine with batching and fallbacks."""

    def __init__(self, model_name: str = "ProsusAI/finbert", lazy_load: bool = True):
        self.model_name = model_name
        self.device = get_optimal_device()
        self.tokenizer = None
        self.model = None
        self.sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
        self.pos_id = 0
        self.neg_id = 1

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Loads model and tokenizer into memory."""
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading {self.model_name} onto {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Inspect label mappings from config
            if hasattr(self.model.config, "id2label"):
                self.sentiment_map = self.model.config.id2label
                label2id = {v.lower(): k for k, v in self.sentiment_map.items()}
                self.pos_id = label2id.get("positive", 0)
                self.neg_id = label2id.get("negative", 1)

            logger.info("FinBERT initialized successfully.")
        except Exception as e:
            logger.warning(
                f"FinBERT weights could not be loaded ({e}). Using financial keyword fallback."
            )
            self.model = None

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Analyzes a list of news headlines.

        Returns:
            sentiments: List of labels ('positive', 'negative', 'neutral')
            confidences: List of softmax confidence scores (0.0 to 1.0)
            scores: Continuous sentiment polarity (-1.0 to +1.0)
        """
        self._load_model()

        if self.model is None:
            return self._fallback_analyze(texts)

        all_sentiments = []
        all_confidences = []
        all_scores = []

        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]
            try:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                confs, preds = torch.max(probs, dim=1)
                polarity = (probs[:, self.pos_id] - probs[:, self.neg_id]).cpu().numpy()

                for p, c, s in zip(preds.cpu().numpy(), confs.cpu().numpy(), polarity):
                    all_sentiments.append(self.sentiment_map.get(int(p), "neutral").lower())
                    all_confidences.append(float(c))
                    all_scores.append(float(s))
            except Exception as e:
                logger.error(f"Batch analysis error at index {i}: {e}")
                for _ in batch:
                    all_sentiments.append("neutral")
                    all_confidences.append(0.5)
                    all_scores.append(0.0)

        return all_sentiments, all_confidences, all_scores

    def _fallback_analyze(
        self, texts: List[str]
    ) -> Tuple[List[str], List[float], List[float]]:
        """Fast financial keyword sentiment scoring when transformer models are unavailable."""
        pos_words = {"surge", "gain", "soar", "jump", "record", "profit", "bull", "rally", "growth", "beat", "high", "upgrade"}
        neg_words = {"slump", "drop", "plunge", "fall", "loss", "bear", "miss", "crash", "down", "risk", "debt", "downgrade"}

        sentiments, confs, scores = [], [], []
        for t in texts:
            words = set(str(t).lower().split())
            p_count = len(words & pos_words)
            n_count = len(words & neg_words)

            if p_count > n_count:
                sentiments.append("positive")
                confs.append(0.8)
                scores.append(min(1.0, 0.3 * (p_count - n_count)))
            elif n_count > p_count:
                sentiments.append("negative")
                confs.append(0.8)
                scores.append(max(-1.0, -0.3 * (n_count - p_count)))
            else:
                sentiments.append("neutral")
                confs.append(0.5)
                scores.append(0.0)
        return sentiments, confs, scores


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates article-level sentiment records into daily time-series features.

    Calculates:
        - Avg_Sentiment: Mean continuous sentiment score (-1.0 to 1.0)
        - News_Count: Total volume of published headlines
        - Sentiment_Ratio: (Positive - Negative) / Total
        - Positive_Count, Negative_Count, Neutral_Count

    Returns:
        DataFrame indexed by Date with daily sentiment metrics.
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame()

    df = news_df.copy()

    # Ensure Date column
    if "Date" not in df.columns:
        if "time_published" in df.columns:
            df["Date"] = pd.to_datetime(df["time_published"]).dt.date
        elif "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"]).dt.date
        else:
            raise KeyError("No valid date column found in news dataframe.")

    df["Date"] = pd.to_datetime(df["Date"])

    # Score column name resolution
    score_col = None
    for cand in ["sentiment_score", "Sentiment_Score", "overall_sentiment_score"]:
        if cand in df.columns:
            score_col = cand
            break

    if score_col is None:
        df["sentiment_score"] = 0.0
        score_col = "sentiment_score"

    # Label column name resolution
    label_col = None
    for cand in ["sentiment_label", "Sentiment", "overall_sentiment_label"]:
        if cand in df.columns:
            label_col = cand
            break

    if label_col:
        df["is_pos"] = df[label_col].astype(str).str.lower().str.contains("pos|bull").astype(int)
        df["is_neg"] = df[label_col].astype(str).str.lower().str.contains("neg|bear").astype(int)
        df["is_neu"] = df[label_col].astype(str).str.lower().str.contains("neu").astype(int)
    else:
        df["is_pos"] = (df[score_col] > 0.1).astype(int)
        df["is_neg"] = (df[score_col] < -0.1).astype(int)
        df["is_neu"] = (df[score_col].abs() <= 0.1).astype(int)

    daily = df.groupby("Date").agg(
        Avg_Sentiment=(score_col, "mean"),
        News_Count=(score_col, "count"),
        Positive_Count=("is_pos", "sum"),
        Negative_Count=("is_neg", "sum"),
        Neutral_Count=("is_neu", "sum"),
    )

    daily["Sentiment_Ratio"] = (daily["Positive_Count"] - daily["Negative_Count"]) / (
        daily["News_Count"] + 1e-6
    )

    return daily
