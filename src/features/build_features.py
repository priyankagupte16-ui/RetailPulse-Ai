"""Feature engineering pipeline for retail demand forecasting.

Builds calendar, lag, and rolling window features strictly shifted
by one day to ensure zero future data leakage.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

# Major US retail holidays for calendar feature enrichment
RETAIL_HOLIDAYS_2023 = {
    "2023-01-01": "New Year's Day",
    "2023-01-16": "MLK Day",
    "2023-02-20": "Presidents' Day",
    "2023-05-29": "Memorial Day",
    "2023-07-04": "Independence Day",
    "2023-09-04": "Labor Day",
    "2023-11-23": "Thanksgiving",
    "2023-11-24": "Black Friday",
    "2023-12-25": "Christmas Day",
    "2023-12-31": "New Year's Eve",
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal calendar signals from sale_date."""
    df = df.copy()
    sale_date = pd.to_datetime(df["sale_date"])

    # Day of week: Monday=0, Sunday=6
    df["day_of_week"] = sale_date.dt.day_of_week
    df["month"] = sale_date.dt.month
    
    # Weekend indicator: Saturday (5) or Sunday (6)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Major holiday indicator
    holiday_dates = set(pd.to_datetime(list(RETAIL_HOLIDAYS_2023.keys())))
    df["is_holiday"] = sale_date.isin(holiday_dates).astype(int)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_days: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Compute lag and rolling window statistics per store and product.

    CRITICAL LEAKAGE RULE:
    Every feature is strictly computed on sales observations shifted by 1 day,
    ensuring day t uses ONLY historical information available up to day t-1.
    """
    if lag_days is None:
        lag_days = [1, 7, 14, 28]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]

    df = df.sort_values(by=["store_id", "product_id", "sale_date"]).copy()
    grouped = df.groupby(["store_id", "product_id"])["quantity_sold"]

    # 1. Base shifted series (yesterday's sales for day t)
    shifted_sales = grouped.shift(1)

    # 2. Lag features (t-1, t-7, t-14, t-28)
    for lag in lag_days:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # 3. Rolling statistics (mean and std) computed strictly on shifted_sales
    # Group shifted_sales by store and product to prevent cross-group boundary leakage
    shifted_grouped = shifted_sales.groupby([df["store_id"], df["product_id"]])

    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = shifted_grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )
        df[f"rolling_std_{window}"] = shifted_grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).std()
        )

    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Execute complete end-to-end feature engineering pipeline."""
    df_features = add_calendar_features(df)
    df_features = add_lag_and_rolling_features(df_features)

    if drop_na:
        # Drop initial 28 days that lack full 28-day historical lag/rolling lookback
        df_features = df_features.dropna().reset_index(drop=True)

    return df_features


if __name__ == "__main__":
    from src.database.queries import load_sales_dataframe

    raw_sales = load_sales_dataframe()
    engineered = build_feature_pipeline(raw_sales, drop_na=False)

    # Select a continuous slice for Store 1, Product 1 showing transition after day 28
    sample_slice = (
        engineered[(engineered["store_id"] == 1) & (engineered["product_id"] == 1)]
        .iloc[28:38]
        .copy()
    )

    display_cols = [
        "sale_date",
        "store_id",
        "product_id",
        "quantity_sold",
        "lag_1",
        "lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ]

    print("\n" + "=" * 90)
    print(" 10-ROW FEATURE ENGINEERING SAMPLE (Store 1, Product 1, Days 29 to 38)")
    print("=" * 90)
    print(sample_slice[display_cols].to_string(index=False))
    print("=" * 90)
