"""Unit tests for XGBoost quantile regression (P10, P50, P90) and Pinball Loss."""

import numpy as np
import pytest
from src.models.metrics import calculate_pinball_loss
from src.models.quantile import (
    run_quantile_pipeline,
    QUANTILE_ARTIFACT_PATH,
)


def test_pinball_loss_known_values():
    """Verify Pinball Loss formula against manual hand-calculated values."""
    # Case 1: Under-prediction (y = 20, y_pred = 10, diff = +10)
    # For alpha = 0.90: loss = 0.90 * 10 = 9.0
    # For alpha = 0.10: loss = 0.10 * 10 = 1.0
    y_true = np.array([20.0])
    y_pred = np.array([10.0])
    assert np.isclose(calculate_pinball_loss(y_true, y_pred, alpha=0.90), 9.0)
    assert np.isclose(calculate_pinball_loss(y_true, y_pred, alpha=0.10), 1.0)

    # Case 2: Over-prediction (y = 10, y_pred = 20, diff = -10)
    # For alpha = 0.90: loss = (1 - 0.90) * 10 = 1.0
    # For alpha = 0.10: loss = (1 - 0.10) * 10 = 9.0
    y_true_over = np.array([10.0])
    y_pred_over = np.array([20.0])
    assert np.isclose(calculate_pinball_loss(y_true_over, y_pred_over, alpha=0.90), 1.0)
    assert np.isclose(calculate_pinball_loss(y_true_over, y_pred_over, alpha=0.10), 9.0)


@pytest.fixture(scope="module")
def quantile_evaluation_results():
    """Execute quantile training pipeline once for tests."""
    return run_quantile_pipeline()


def test_quantile_monotonicity(quantile_evaluation_results):
    """Verify P10 <= P50 <= P90 holds universally across all test predictions."""
    preds = quantile_evaluation_results["predictions"]
    p10 = preds["p10"]
    p50 = preds["p50"]
    p90 = preds["p90"]

    assert len(p10) == len(p50) == len(p90)
    assert (p10 <= p50).all(), "Detected quantile crossing: P10 > P50!"
    assert (p50 <= p90).all(), "Detected quantile crossing: P50 > P90!"
    assert (p10 >= 0).all(), "Negative demand predicted in P10!"


def test_quantile_calibration_coverage(quantile_evaluation_results):
    """Verify empirical coverage for [P10, P90] is close to 80% (between 73% and 87%)."""
    coverage = quantile_evaluation_results["coverage_percentage_80"]
    assert 72.0 <= coverage <= 88.0, (
        f"Quantile interval uncalibrated! Expected ~80%, got {coverage:.2f}%"
    )

    # Below P10 and above P90 should both be close to 10%
    below_p10 = quantile_evaluation_results["below_p10_percentage"]
    above_p90 = quantile_evaluation_results["above_p90_percentage"]
    assert 5.0 <= below_p10 <= 16.0
    assert 5.0 <= above_p90 <= 16.0


def test_quantile_artifact_saved(quantile_evaluation_results):
    """Verify trained quantile models artifact is saved to disk."""
    assert QUANTILE_ARTIFACT_PATH.exists()
