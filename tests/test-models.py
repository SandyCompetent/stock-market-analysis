"""Unit tests for models and evaluation metrics."""

import numpy as np
import pytest

from src.models.baselines import NaivePredictor, SVMPredictor
from src.models.evaluation import calculate_metrics, compare_models
from src.models.lstm import LSTMTrainer, StockLSTM


def test_stock_lstm_forward():
    batch_size = 16
    seq_len = 30
    input_dim = 7

    model = StockLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
    dummy_input = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

    import torch

    output = model(torch.tensor(dummy_input))
    assert output.shape == (batch_size,)


def test_svm_predictor():
    X = np.random.randn(50, 10, 4)
    y = np.random.randn(50)

    svm = SVMPredictor()
    svm.fit(X, y)
    preds = svm.predict(X)

    assert preds.shape == (50,)


def test_naive_predictor():
    series = np.array([10.0, 12.0, 15.0, 14.0])
    preds = NaivePredictor.predict(series)
    assert preds[0] == 10.0
    assert preds[1] == 10.0
    assert preds[2] == 12.0
    assert preds[3] == 15.0


def test_calculate_metrics():
    actual = np.array([10.0, 12.0, 14.0, 16.0])
    predicted = np.array([10.5, 11.5, 14.2, 15.8])

    metrics = calculate_metrics(actual, predicted, is_percentage=False)
    assert "RMSE" in metrics
    assert "MAE" in metrics
    assert "MAPE (%)" in metrics
    assert "Directional Accuracy (%)" in metrics
    assert metrics["RMSE"] > 0
    assert metrics["Directional Accuracy (%)"] == 100.0


def test_compare_models():
    actual = np.array([10.0, 12.0, 14.0, 16.0])
    preds = {
        "Model A": np.array([10.1, 11.9, 14.1, 15.9]),
        "Model B": np.array([11.0, 13.0, 15.0, 17.0]),
    }
    df = compare_models(preds, actual)
    assert len(df) == 2
    assert "Rank" in df.columns
    assert df.loc["Model A", "Rank"] == 1
