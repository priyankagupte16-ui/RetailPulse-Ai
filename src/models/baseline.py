"""Baseline demand forecasting models and backtest evaluation.

Implements 7-Day Moving Average and Seasonal Naive (Same Day Last Week)
evaluated on a strictly out-of-sample 28-day holdout set.
"""

from datetime import timedelta
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.database.queries import load_sales_dataframe
from src.models.metrics import calculate_mae, calculate_rmse, calculate_wape


def forecast_moving_average(history: np.ndarray, window: int = 7, horizon: int = 28) -> np.ndarray:
    """Generate a constant moving average forecast across the horizon."""
    if len(history) < window:
        avg_value = float(np.mean(history)) if len(history) > 0 else 0.0
    else:
        avg_value = float(np.mean(history[-window:]))
    return np.full(shape=horizon, fill_value=avg_value, dtype=float)


def forecast_seasonal_naive(history: np.ndarray, seasonality: int = 7, horizon: int = 28) -> np.ndarray:
    """Generate a seasonal naive forecast repeating the last seasonal cycle."""
    if len(history) < seasonality:
        return forecast_moving_average(history, window=len(history), horizon=horizon)
    
    # Grab the last `seasonality` days from training history
    last_cycle = history[-seasonality:]
    
    # Tile the cycle to cover the full horizon
    repetitions = int(np.ceil(horizon / seasonality))
    tiled = np.tile(last_cycle, repetitions)
    return tiled[:horizon].astype(float)


def evaluate_baselines(test_days: int = 28) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate both baseline methods across all store-product time series."""
    df = load_sales_dataframe()
    max_date = df["sale_date"].max()
    cutoff_date = max_date - timedelta(days=test_days)

    train_df = df[df["sale_date"] <= cutoff_date].copy()
    test_df = df[df["sale_date"] > cutoff_date].copy()

    unique_series = train_df[["store_id", "product_id"]].drop_duplicates().values

    results = []

    for store_id, product_id in unique_series:
        s_train = train_df[
            (train_df["store_id"] == store_id) & (train_df["product_id"] == product_id)
        ].sort_values("sale_date")["quantity_sold"].to_numpy()

        s_test = test_df[
            (test_df["store_id"] == store_id) & (test_df["product_id"] == product_id)
        ].sort_values("sale_date")["quantity_sold"].to_numpy()

        horizon = len(s_test)
        if horizon == 0:
            continue

        pred_ma = forecast_moving_average(s_train, window=7, horizon=horizon)
        pred_sn = forecast_seasonal_naive(s_train, seasonality=7, horizon=horizon)

        for i in range(horizon):
            results.append({
                "store_id": store_id,
                "product_id": product_id,
                "step": i + 1,
                "actual": s_test[i],
                "pred_moving_average": pred_ma[i],
                "pred_seasonal_naive": pred_sn[i],
            })

    results_df = pd.DataFrame(results)

    # Calculate overall aggregate metrics
    y_true = results_df["actual"].to_numpy()
    y_pred_ma = results_df["pred_moving_average"].to_numpy()
    y_pred_sn = results_df["pred_seasonal_naive"].to_numpy()

    metrics = {
        "evaluation_period_days": test_days,
        "total_series_evaluated": len(unique_series),
        "total_test_observations": len(y_true),
        "moving_average_7d": {
            "wape": calculate_wape(y_true, y_pred_ma),
            "mae": calculate_mae(y_true, y_pred_ma),
            "rmse": calculate_rmse(y_true, y_pred_ma),
        },
        "seasonal_naive_7d": {
            "wape": calculate_wape(y_true, y_pred_sn),
            "mae": calculate_mae(y_true, y_pred_sn),
            "rmse": calculate_rmse(y_true, y_pred_sn),
        },
    }

    return metrics, results_df


if __name__ == "__main__":
    metrics_summary, _ = evaluate_baselines(test_days=28)
    
    print("\n" + "=" * 55)
    print(" BASELINE MODEL PERFORMANCE REPORT (Last 28 Days Holdout)")
    print("=" * 55)
    print(f"Total Series: {metrics_summary['total_series_evaluated']}")
    print(f"Total Test Observations: {metrics_summary['total_test_observations']:,}")
    print("-" * 55)
    
    ma = metrics_summary["moving_average_7d"]
    print(f"1. Moving Average (7-Day Window):")
    print(f"   - WAPE : {ma['wape'] * 100:.2f}%")
    print(f"   - MAE  : {ma['mae']:.2f} units")
    print(f"   - RMSE : {ma['rmse']:.2f} units")
    
    print("-" * 55)
    sn = metrics_summary["seasonal_naive_7d"]
    print(f"2. Seasonal Naive (Same Day Last Week):")
    print(f"   - WAPE : {sn['wape'] * 100:.2f}%")
    print(f"   - MAE  : {sn['mae']:.2f} units")
    print(f"   - RMSE : {sn['rmse']:.2f} units")
    print("=" * 55)
