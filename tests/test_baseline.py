"""Unit tests for baseline forecasting models and evaluation metrics."""

import numpy as np
import pytest
from src.models.metrics import calculate_mae, calculate_rmse, calculate_wape
from src.models.baseline import (
    forecast_moving_average,
    forecast_seasonal_naive,
    evaluate_baselines,
)


def test_metrics_with_known_vectors():
    """Verify MAE, RMSE, and WAPE with known mathematical values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])

    # Absolute errors: [2, 2, 3, 4] -> sum = 11, mean = 2.75
    assert calculate_mae(y_true, y_pred) == 2.75

    # Squared errors: [4, 4, 9, 16] -> mean = 8.25 -> sqrt = 2.87228
    assert np.isclose(calculate_rmse(y_true, y_pred), np.sqrt(8.25))

    # Total actual = 100. WAPE = 11 / 100 = 0.11 (11%)
    assert calculate_wape(y_true, y_pred) == 0.11


def test_metrics_zero_actual_edge_case():
    """Verify WAPE gracefully returns 0.0 if all actuals are zero."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert calculate_wape(y_true, y_pred) == 0.0


def test_forecast_moving_average_logic():
    """Verify moving average calculates average of last N items and tiles it."""
    history = np.array([10, 20, 30, 40, 50, 60, 70])  # sum = 280, mean = 40
    pred = forecast_moving_average(history, window=7, horizon=14)

    assert len(pred) == 14
    assert np.all(pred == 40.0)


def test_forecast_seasonal_naive_logic():
    """Verify seasonal naive repeats the exact last 7-day pattern across the horizon."""
    # Last 7 days: [1, 2, 3, 4, 5, 6, 7]
    history = np.array([100, 200, 1, 2, 3, 4, 5, 6, 7])
    pred = forecast_seasonal_naive(history, seasonality=7, horizon=14)

    assert len(pred) == 14
    # Week 1
    assert np.array_equal(pred[:7], np.array([1, 2, 3, 4, 5, 6, 7]))
    # Week 2 (repeating cycle)
    assert np.array_equal(pred[7:14], np.array([1, 2, 3, 4, 5, 6, 7]))


def test_evaluate_baselines_contract():
    """Verify evaluation runs across all 75 series and outputs valid non-empty metrics."""
    metrics, results_df = evaluate_baselines(test_days=28)

    assert metrics["evaluation_period_days"] == 28
    assert metrics["total_series_evaluated"] == 75
    assert metrics["total_test_observations"] == 75 * 28

    ma = metrics["moving_average_7d"]
    sn = metrics["seasonal_naive_7d"]

    assert 0.0 < ma["wape"] < 1.0
    assert ma["mae"] > 0
    assert ma["rmse"] > 0

    assert 0.0 < sn["wape"] < 1.0
    assert sn["mae"] > 0
    assert sn["rmse"] > 0

    assert len(results_df) == 75 * 28
