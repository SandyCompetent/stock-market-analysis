"""Performance evaluation metrics and model comparison utilities."""

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    is_percentage: bool = False,
) -> Dict[str, float]:
    """Calculates forecasting evaluation metrics.

    Metrics:
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - MAPE: Mean Absolute Percentage Error (%)
        - Directional_Accuracy: Percentage of correctly predicted price movement directions (%)
    """
    y_true = np.asarray(actual).flatten()
    y_pred = np.asarray(predicted).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch between actual ({len(y_true)}) and predicted ({len(y_pred)})."
        )

    # 1. Error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 2. MAPE (avoid division by zero)
    denom = np.where(np.abs(y_true) < 1e-4, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    # 3. Directional Accuracy
    if is_percentage:
        # In percentage change target mode, sign directly indicates direction
        true_dir = y_true > 0
        pred_dir = y_pred > 0
    else:
        # In price target mode, compare step-over-step diff
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0

    directional_acc = (
        float(np.mean(true_dir == pred_dir) * 100.0) if len(true_dir) > 0 else 50.0
    )

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 2),
        "Directional Accuracy (%)": round(directional_acc, 2),
    }


def compare_models(
    results_dict: Dict[str, np.ndarray],
    actual: np.ndarray,
    is_percentage: bool = False,
) -> pd.DataFrame:
    """Generates a performance leaderboard comparing multiple candidate models.

    Args:
        results_dict: Dictionary mapping model names to prediction arrays.
        actual: Ground-truth target array.
        is_percentage: Whether target is percentage change or raw price.

    Returns:
        Sorted summary DataFrame with ranks.
    """
    rows = []
    for model_name, preds in results_dict.items():
        metrics = calculate_metrics(actual, preds, is_percentage=is_percentage)
        metrics["Model"] = model_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df.set_index("Model", inplace=True)

    # Rank primarily by RMSE ascending, then Directional Accuracy descending
    df.sort_values(by=["RMSE", "Directional Accuracy (%)"], ascending=[True, False], inplace=True)
    df["Rank"] = range(1, len(df) + 1)

    cols = ["Rank", "RMSE", "MAE", "MAPE (%)", "Directional Accuracy (%)"]
    return df[cols]
