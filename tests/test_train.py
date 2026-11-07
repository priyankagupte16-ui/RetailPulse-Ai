"""Unit tests for XGBoost model training, temporal split, and evaluation."""

import numpy as np
import pandas as pd
import pytest
from src.models.train import (
    prepare_datasets,
    train_xgboost,
    evaluate_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_ARTIFACT_PATH,
)


@pytest.fixture(scope="module")
def prepared_data():
    """Prepare train and test partitions once for testing."""
    return prepare_datasets(test_days=28)


def test_prepare_datasets_strict_temporal_split(prepared_data):
    """Verify train and test partitions do not overlap and are strictly split by time."""
    X_train, y_train, X_test, y_test, test_df = prepared_data

    assert len(X_train) > 0
    assert len(X_test) == 75 * 28  # 75 series * 28 days = 2,100 observations
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Date integrity: Test set must only contain dates in the final 28 days
    assert (test_df["sale_date"] >= "2023-12-04").all()


def test_no_target_in_features(prepared_data):
    """Verify target column quantity_sold is never present in input features."""
    X_train, _, X_test, _, _ = prepared_data

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
    assert set(FEATURE_COLUMNS).issubset(set(X_train.columns))


def test_model_training_and_non_negative_predictions(prepared_data):
    """Verify model fits, predicts, and guarantees non-negative demand outputs."""
    X_train, y_train, X_test, y_test, _ = prepared_data

    # Train a fast lightweight model for unit testing
    model = train_xgboost(X_train, y_train, n_estimators=30, max_depth=3)
    results = evaluate_model(model, X_test, y_test)

    preds = results["predictions"]
    assert len(preds) == len(X_test)
    assert (preds >= 0).all(), "Predictions must not be negative in retail!"
    assert 0.0 < results["wape"] < 1.0
    assert results["mae"] > 0
    assert results["rmse"] > 0


def test_xgboost_beats_baseline(prepared_data):
    """Verify full XGBoost model outperforms the 23.97% baseline WAPE."""
    X_train, y_train, X_test, y_test, _ = prepared_data

    model = train_xgboost(X_train, y_train, n_estimators=100, max_depth=5)
    results = evaluate_model(model, X_test, y_test)

    # Baseline WAPE was 23.97%
    assert results["wape"] < 0.2397, (
        f"XGBoost WAPE ({results['wape']*100:.2f}%) did not beat "
        f"baseline benchmark (23.97%)!"
    )
