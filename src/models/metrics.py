"""Evaluation metrics for time-series demand forecasting."""

import numpy as np


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    Formula: sum(|y_true - y_pred|) / sum(y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual)


def calculate_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Calculate Pinball (Quantile) Loss for a given target quantile alpha in (0, 1).

    Formula: mean(max(alpha * (y - y_hat), (alpha - 1) * (y - y_hat)))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))

