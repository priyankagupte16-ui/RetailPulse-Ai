"""Pytest suite proving zero data leakage in feature engineering.

Mathematically verifies:
1. Altering sales on day t does NOT change any feature on day t.
2. The initial rows have NaNs for lags and are NEVER filled with future values (no bfill).
3. Features for one store/product are completely isolated and unaffected by other stores.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_calendar_features,
    add_lag_and_rolling_features,
    build_feature_pipeline,
)


@pytest.fixture
def multi_store_sales_df():
    """Create a realistic 45-day sales dataset for Store 1 and Store 2."""
    dates = pd.date_range("2023-01-01", periods=45, freq="D")
    records = []
    for store_id in [1, 2]:
        for i, d in enumerate(dates):
            # Store 1 has base 50, Store 2 has base 200
            base = 50 if store_id == 1 else 200
            records.append({
                "sale_date": d,
                "store_id": store_id,
                "product_id": 1,
                "quantity_sold": base + (i % 7),
                "revenue": float((base + (i % 7)) * 3.50),
                "on_promotion": (i % 10 == 0),
            })
    return pd.DataFrame(records)


def test_proof_1_changing_day_t_sales_does_not_change_any_day_t_feature(multi_store_sales_df):
    """PROOF 1: Modifying quantity_sold on day t must NOT alter ANY feature on day t.

    If any feature on day t changes when day t's sales change, the feature is
    leaking the target value.
    """
    feature_cols = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
        "day_of_week", "month", "is_weekend", "is_holiday",
    ]

    # Baseline feature extraction
    df_original = build_feature_pipeline(multi_store_sales_df.copy(), drop_na=False)

    # Create a tampered copy where day 35 sales is spiked by 1,000,000 units
    df_tampered = multi_store_sales_df.copy()
    target_mask = (df_tampered["store_id"] == 1) & (df_tampered["sale_date"] == "2023-02-05")
    df_tampered.loc[target_mask, "quantity_sold"] = 1_000_000

    df_tampered_features = build_feature_pipeline(df_tampered, drop_na=False)

    row_orig = df_original[target_mask].iloc[0]
    row_tampered = df_tampered_features[target_mask].iloc[0]

    # Verify every single feature on day t is 100% identical
    for col in feature_cols:
        val_orig = row_orig[col]
        val_tampered = row_tampered[col]

        if pd.isna(val_orig):
            assert pd.isna(val_tampered), f"Feature {col} diverged on NaN!"
        else:
            assert np.isclose(val_orig, val_tampered), (
                f"DATA LEAKAGE DETECTED in '{col}'! "
                f"Original: {val_orig}, Tampered: {val_tampered}. "
                f"Feature changed when day t sales was altered!"
            )


def test_proof_2_first_rows_have_nans_and_no_future_backfill(multi_store_sales_df):
    """PROOF 2: Initial rows have NaNs for lags and are NEVER filled with future values.

    If lag_1 has a number on row 0, or lag_28 has numbers on row 5,
    future values have leaked backward (e.g., via bfill).
    """
    df_features = build_feature_pipeline(multi_store_sales_df.copy(), drop_na=False)

    # Inspect Store 1 sorted chronologically
    s1 = df_features[df_features["store_id"] == 1].sort_values("sale_date").reset_index(drop=True)

    # Row 0 (Day 1) cannot have lag_1, lag_7, lag_14, lag_28, or rolling statistics
    assert pd.isna(s1.loc[0, "lag_1"]), "Row 0 lag_1 must be NaN (no day 0 exists)"
    assert pd.isna(s1.loc[0, "lag_7"]), "Row 0 lag_7 must be NaN"
    assert pd.isna(s1.loc[0, "rolling_mean_7"]), "Row 0 rolling_mean_7 must be NaN"

    # Exactly the first 1 row of lag_1 must be NaN
    assert s1["lag_1"].isna().sum() == 1

    # Exactly the first 7 rows of lag_7 must be NaN (indices 0 to 6)
    assert s1["lag_7"].isna().sum() == 7
    assert s1.loc[0:6, "lag_7"].isna().all()
    assert not pd.isna(s1.loc[7, "lag_7"]), "Day 8 should have valid lag_7 from Day 1"

    # Exactly the first 14 rows of lag_14 must be NaN (indices 0 to 13)
    assert s1["lag_14"].isna().sum() == 14
    assert s1.loc[0:13, "lag_14"].isna().all()

    # Exactly the first 28 rows of lag_28 must be NaN (indices 0 to 27)
    assert s1["lag_28"].isna().sum() == 28
    assert s1.loc[0:27, "lag_28"].isna().all()
    assert not pd.isna(s1.loc[28, "lag_28"]), "Day 29 should have valid lag_28 from Day 1"

    # Exactly the first 7 rows of rolling_mean_7 must be NaN (because min_periods=7 on shifted data)
    assert s1.loc[0:6, "rolling_mean_7"].isna().all()
    assert not pd.isna(s1.loc[7, "rolling_mean_7"])


def test_proof_3_store_and_product_group_complete_isolation(multi_store_sales_df):
    """PROOF 3: Store 1 features are identical whether processed alone or alongside other stores.

    Guarantees that rolling and lag computations per store-product never bleed across boundaries.
    """
    # 1. Process Store 1 alongside Store 2 in the full dataset
    df_combined_features = build_feature_pipeline(multi_store_sales_df.copy(), drop_na=False)
    s1_from_combined = (
        df_combined_features[df_combined_features["store_id"] == 1]
        .sort_values("sale_date")
        .reset_index(drop=True)
    )

    # 2. Process Store 1 completely isolated on its own
    df_store1_only = multi_store_sales_df[multi_store_sales_df["store_id"] == 1].copy()
    s1_isolated_features = (
        build_feature_pipeline(df_store1_only, drop_na=False)
        .sort_values("sale_date")
        .reset_index(drop=True)
    )

    feature_cols = [
        "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
    ]

    for col in feature_cols:
        combined_vals = s1_from_combined[col]
        isolated_vals = s1_isolated_features[col]
        
        # Check equality including NaNs matching in the exact same positions
        pd.testing.assert_series_equal(
            combined_vals,
            isolated_vals,
            check_names=False,
            obj=f"Feature {col} mismatch between combined and isolated runs"
        )
