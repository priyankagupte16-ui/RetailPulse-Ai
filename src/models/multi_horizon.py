"""Direct multi-horizon demand forecasting engine (1 to 14 days ahead).

Trains 14 independent XGBoost models to eliminate recursive error compounding.
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.database.queries import load_sales_dataframe
from src.features.build_features import build_feature_pipeline
from src.models.metrics import calculate_wape, calculate_mae, calculate_rmse

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MULTI_HORIZON_ARTIFACT_PATH = MODEL_DIR / "direct_14d_models.joblib"

# Shared feature columns known at forecast origin
BASE_FEATURE_COLS = [
    "store_id",
    "product_id",
    "unit_price",
    "unit_cost",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]


class DirectMultiHorizonForecaster:
    """Direct multi-step forecaster managing 14 horizon-specific models."""

    def __init__(self, max_horizon: int = 14):
        self.max_horizon = max_horizon
        self.models: Dict[int, XGBRegressor] = {}
        self.feature_cols_by_horizon: Dict[int, List[str]] = {}

    def _prepare_horizon_dataset(
        self, df_features: pd.DataFrame, horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align origin features at time t with target and calendar signals at time t+h."""
        df = df_features.sort_values(["store_id", "product_id", "sale_date"]).copy()
        grouped = df.groupby(["store_id", "product_id"])

        # Target is sales shifted backward by h steps (looking h steps into the future)
        df["target"] = grouped["quantity_sold"].shift(-horizon)

        # Future calendar & promotion signals known in advance for day t+h
        future_date = df["sale_date"] + timedelta(days=horizon)
        df["target_day_of_week"] = future_date.dt.day_of_week
        df["target_is_weekend"] = df["target_day_of_week"].isin([5, 6]).astype(int)
        df["target_month"] = future_date.dt.month
        
        # Future scheduled promotion for day t+h
        df["target_on_promotion"] = grouped["on_promotion"].shift(-horizon).fillna(False).astype(int)

        feature_cols = BASE_FEATURE_COLS + [
            "target_day_of_week",
            "target_is_weekend",
            "target_month",
            "target_on_promotion",
        ]

        # Drop rows where target or features are NaN
        valid_df = df.dropna(subset=["target"] + feature_cols).copy()
        return valid_df[feature_cols], valid_df["target"], valid_df["sale_date"]

    def fit(self, df_features: pd.DataFrame, test_days: int = 14) -> Dict[str, Any]:
        """Train 14 independent direct models on historical timeline."""
        max_date = df_features["sale_date"].max()
        cutoff_date = max_date - timedelta(days=test_days)

        for h in range(1, self.max_horizon + 1):
            X, y, dates = self._prepare_horizon_dataset(df_features, horizon=h)
            
            # Strict temporal train partition
            train_mask = (dates + timedelta(days=h)) <= cutoff_date
            X_train = X[train_mask]
            y_train = y[train_mask]

            model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42 + h,
                n_jobs=-1,
                objective="reg:squarederror",
            )
            model.fit(X_train, y_train)
            self.models[h] = model
            self.feature_cols_by_horizon[h] = list(X.columns)

        return {"models_trained": len(self.models)}

    def evaluate(self, df_features: pd.DataFrame, test_days: int = 14) -> pd.DataFrame:
        """Evaluate each horizon day independently on the holdout test period."""
        max_date = df_features["sale_date"].max()
        cutoff_date = max_date - timedelta(days=test_days)

        horizon_metrics = []

        for h in range(1, self.max_horizon + 1):
            X, y, dates = self._prepare_horizon_dataset(df_features, horizon=h)
            
            # Test partition: rows where target date falls strictly in the test period
            target_dates = dates + timedelta(days=h)
            test_mask = (target_dates > cutoff_date) & (target_dates <= max_date)

            X_test = X[test_mask]
            y_test = y[test_mask].to_numpy()

            model = self.models[h]
            y_pred_raw = model.predict(X_test)
            y_pred = np.clip(y_pred_raw, 0, None)

            wape = calculate_wape(y_test, y_pred)
            mae = calculate_mae(y_test, y_pred)
            rmse = calculate_rmse(y_test, y_pred)

            horizon_metrics.append({
                "horizon_day": h,
                "wape": wape,
                "mae": mae,
                "rmse": rmse,
                "test_observations": len(y_test),
            })

        return pd.DataFrame(horizon_metrics)

    def save(self, filepath: Path = MULTI_HORIZON_ARTIFACT_PATH) -> None:
        """Persist all 14 models to disk."""
        joblib.dump(self, filepath)


def train_and_evaluate_multi_horizon() -> Tuple[DirectMultiHorizonForecaster, pd.DataFrame]:
    """Execute complete 14-day direct forecasting pipeline."""
    raw_df = load_sales_dataframe()
    features_df = build_feature_pipeline(raw_df, drop_na=True)

    forecaster = DirectMultiHorizonForecaster(max_horizon=14)
    forecaster.fit(features_df, test_days=14)
    metrics_df = forecaster.evaluate(features_df, test_days=14)
    forecaster.save()

    return forecaster, metrics_df


if __name__ == "__main__":
    _, metrics_df = train_and_evaluate_multi_horizon()

    print("\n" + "=" * 70)
    print(" DIRECT MULTI-HORIZON EVALUATION (14 Days Ahead: WAPE, MAE, RMSE)")
    print("=" * 70)
    print(f"{'Horizon':<12} | {'Forecast Target':<18} | {'WAPE':<10} | {'MAE (units)':<12} | {'RMSE':<10}")
    print("-" * 70)
    for _, row in metrics_df.iterrows():
        h = int(row["horizon_day"])
        print(f"Day {h:<8} | t + {h:<14} | {row['wape']*100:>8.2f}% | {row['mae']:>12.2f} | {row['rmse']:>10.2f}")
    print("=" * 70)

    avg_wape = metrics_df["wape"].mean() * 100
    day1_wape = metrics_df.loc[metrics_df["horizon_day"] == 1, "wape"].values[0] * 100
    day14_wape = metrics_df.loc[metrics_df["horizon_day"] == 14, "wape"].values[0] * 100
    print(f"\n[INSIGHT] Day 1 WAPE: {day1_wape:.2f}%  -->  Day 14 WAPE: {day14_wape:.2f}% (Average 14-Day WAPE: {avg_wape:.2f}%)")
