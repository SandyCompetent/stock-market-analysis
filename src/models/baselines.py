"""Baseline forecasting models (Support Vector Regression, ARIMA, and Naive persistence)."""

import logging
from typing import Optional, Union

import numpy as np
from sklearn.svm import SVR

logger = logging.getLogger(__name__)


class SVMPredictor:
    """Support Vector Machine for regression on flattened time-series sequences."""

    def __init__(
        self,
        C: float = 100.0,
        gamma: Union[str, float] = "scale",
        epsilon: float = 0.05,
    ):
        self.model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)

    def _flatten_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            nsamples, nx, ny = X.shape
            return X.reshape((nsamples, nx * ny))
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._flatten_input(X)
        logger.info(f"Training SVR model on {X_flat.shape} features...")
        self.model.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten_input(X)
        return self.model.predict(X_flat)


class ARIMAPredictor:
    """Autoregressive Integrated Moving Average (ARIMA) statistical model."""

    def __init__(self, order: tuple = (5, 1, 0)):
        self.order = order
        self.fitted_model = None

    def fit(self, series: np.ndarray):
        """Fits ARIMA to historical 1D series."""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            logger.info(f"Fitting ARIMA{self.order} model...")
            model = ARIMA(series, order=self.order)
            self.fitted_model = model.fit()
        except ImportError:
            logger.warning(
                "statsmodels not installed. ARIMA will use linear extrapolation fallback."
            )
            self.fitted_model = None
        except Exception as e:
            logger.warning(f"ARIMA fit failed ({e}). Using linear trend fallback.")
            self.fitted_model = None
        return self

    def forecast(
        self, steps: int, last_history: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forecasts the next n steps."""
        if self.fitted_model is not None:
            try:
                preds = self.fitted_model.forecast(steps=steps)
                return np.asarray(preds)
            except Exception as e:
                logger.error(f"ARIMA forecast error: {e}")

        # Fallback extrapolation
        if last_history is not None and len(last_history) >= 2:
            trend = last_history[-1] - last_history[-2]
            return np.array(
                [last_history[-1] + (i + 1) * trend * 0.5 for i in range(steps)]
            )
        return np.zeros(steps)


class NaivePredictor:
    """Naive persistence benchmark (today's prediction = yesterday's value)."""

    @staticmethod
    def predict(series: np.ndarray) -> np.ndarray:
        """Returns shifted series."""
        preds = np.roll(series, 1)
        preds[0] = series[0]
        return preds
