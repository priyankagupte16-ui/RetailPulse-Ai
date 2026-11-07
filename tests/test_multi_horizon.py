"""Unit tests for direct multi-horizon forecasting (1 to 14 days ahead)."""

import pytest
import pandas as pd
from src.models.multi_horizon import (
    DirectMultiHorizonForecaster,
    train_and_evaluate_multi_horizon,
    MULTI_HORIZON_ARTIFACT_PATH,
)


@pytest.fixture(scope="module")
def multi_horizon_results():
    """Execute multi-horizon training and evaluation once for tests."""
    forecaster, metrics_df = train_and_evaluate_multi_horizon()
    return forecaster, metrics_df


def test_multi_horizon_models_count(multi_horizon_results):
    """Verify exactly 14 independent direct models are trained and saved."""
    forecaster, _ = multi_horizon_results

    assert len(forecaster.models) == 14
    for h in range(1, 15):
        assert h in forecaster.models, f"Model for horizon {h} missing!"
    assert MULTI_HORIZON_ARTIFACT_PATH.exists()


def test_multi_horizon_evaluation_structure(multi_horizon_results):
    """Verify all 14 horizon days are evaluated with valid positive metrics."""
    _, metrics_df = multi_horizon_results

    assert len(metrics_df) == 14
    assert set(metrics_df["horizon_day"]) == set(range(1, 15))

    for _, row in metrics_df.iterrows():
        assert 0.0 < row["wape"] < 1.0
        assert row["mae"] > 0
        assert row["rmse"] > 0
        assert row["test_observations"] > 0


def test_error_growth_later_days(multi_horizon_results):
    """Verify that forecasting further out generally incurs higher error than day 1."""
    _, metrics_df = multi_horizon_results

    wape_day1 = metrics_df.loc[metrics_df["horizon_day"] == 1, "wape"].values[0]
    wape_day14 = metrics_df.loc[metrics_df["horizon_day"] == 14, "wape"].values[0]

    # Day 14 error should exceed Day 1 error due to information decay
    assert wape_day14 > wape_day1, (
        f"Expected Day 14 WAPE ({wape_day14:.4f}) to be greater than Day 1 WAPE ({wape_day1:.4f})"
    )
