"""Quantile demand forecasting models (P10, P50, P90) using Pinball Loss.

Enables uncertainty quantification and risk-aware inventory replenishment.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.train import prepare_datasets, FEATURE_COLUMNS
from src.models.metrics import calculate_pinball_loss, calculate_mae, calculate_wape

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
QUANTILE_ARTIFACT_PATH = MODEL_DIR / "quantile_models.joblib"


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    n_estimators: int = 120,
    max_depth: int = 4,
    learning_rate: float = 0.08,
) -> XGBRegressor:
    """Train a single XGBoost regressor optimizing for quantile (pinball) loss."""
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=int(42 + alpha * 100),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Tuple[float, ...] = (0.10, 0.50, 0.90),
) -> Dict[str, XGBRegressor]:
    """Train P10, P50, and P90 quantile regression models."""
    models = {}
    for alpha in alphas:
        label = f"P{int(alpha * 100)}"
        models[label] = train_quantile_model(X_train, y_train, alpha=alpha)
    return models


def evaluate_quantiles(
    models: Dict[str, XGBRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate quantile predictions, pinball losses, and prediction interval calibration."""
    y_true = y_test.to_numpy()

    # Raw predictions clipped at 0 (non-negative retail demand)
    p10_raw = np.clip(models["P10"].predict(X_test), 0, None)
    p50_raw = np.clip(models["P50"].predict(X_test), 0, None)
    p90_raw = np.clip(models["P90"].predict(X_test), 0, None)

    # Monotonic sorting to prevent quantile crossing (ensuring P10 <= P50 <= P90)
    stacked = np.vstack([p10_raw, p50_raw, p90_raw])
    stacked_sorted = np.sort(stacked, axis=0)

    p10 = stacked_sorted[0]
    p50 = stacked_sorted[1]
    p90 = stacked_sorted[2]

    # Calculate Pinball Loss per quantile
    pinball_p10 = calculate_pinball_loss(y_true, p10, alpha=0.10)
    pinball_p50 = calculate_pinball_loss(y_true, p50, alpha=0.50)
    pinball_p90 = calculate_pinball_loss(y_true, p90, alpha=0.90)
    mean_pinball = (pinball_p10 + pinball_p50 + pinball_p90) / 3.0

    # Calibration Check: Empirical Coverage (What % of actuals fall between P10 and P90?)
    # Theoretical expectation for [P10, P90] interval is exactly 90% - 10% = 80.0%
    inside_interval = (y_true >= p10) & (y_true <= p90)
    coverage_percentage = float(np.mean(inside_interval) * 100)

    # Percentage falling below P10 (expected ~10%) and above P90 (expected ~10%)
    below_p10_pct = float(np.mean(y_true < p10) * 100)
    above_p90_pct = float(np.mean(y_true > p90) * 100)

    # Average Interval Width (Measures sharpness / uncertainty spread)
    avg_interval_width = float(np.mean(p90 - p10))

    return {
        "pinball_loss_p10": pinball_p10,
        "pinball_loss_p50": pinball_p50,
        "pinball_loss_p90": pinball_p90,
        "mean_pinball_loss": mean_pinball,
        "coverage_percentage_80": coverage_percentage,
        "below_p10_percentage": below_p10_pct,
        "above_p90_percentage": above_p90_pct,
        "avg_interval_width": avg_interval_width,
        "p50_wape": calculate_wape(y_true, p50),
        "p50_mae": calculate_mae(y_true, p50),
        "predictions": {
            "p10": p10,
            "p50": p50,
            "p90": p90,
        },
    }


def save_quantile_models(models: Dict[str, XGBRegressor], filepath: Path = QUANTILE_ARTIFACT_PATH) -> None:
    """Save trained quantile models dictionary to disk."""
    joblib.dump(models, filepath)


def run_quantile_pipeline() -> Dict[str, Any]:
    """Execute complete quantile training, evaluation, and serialization."""
    X_train, y_train, X_test, y_test, _ = prepare_datasets(test_days=28)
    models = train_all_quantiles(X_train, y_train)
    metrics = evaluate_quantiles(models, X_test, y_test)
    save_quantile_models(models)
    return metrics


if __name__ == "__main__":
    metrics = run_quantile_pipeline()

    print("\n" + "=" * 70)
    print(" XGBOOST QUANTILE DEMAND FORECASTING (P10, P50, P90)")
    print("=" * 70)
    print("1. PINBALL (QUANTILE) LOSS:")
    print(f"   - P10 Loss (alpha = 0.10) : {metrics['pinball_loss_p10']:.4f}")
    print(f"   - P50 Loss (alpha = 0.50) : {metrics['pinball_loss_p50']:.4f}  (Median MAE proxy: {metrics['p50_mae']:.2f} units)")
    print(f"   - P90 Loss (alpha = 0.90) : {metrics['pinball_loss_p90']:.4f}")
    print(f"   - Mean Pinball Loss       : {metrics['mean_pinball_loss']:.4f}")
    print("-" * 70)
    print("2. CALIBRATION & COVERAGE EVALUATION ([P10, P90] Interval):")
    print(f"   - Expected Coverage       : 80.00%")
    print(f"   - Actual Empirical Coverage: {metrics['coverage_percentage_80']:.2f}%")
    print(f"   - Actuals below P10       : {metrics['below_p10_percentage']:.2f}% (Expected ~10%)")
    print(f"   - Actuals above P90 (Peaks): {metrics['above_p90_percentage']:.2f}% (Expected ~10%)")
    print(f"   - Average Interval Spread : {metrics['avg_interval_width']:.2f} units")
    print("=" * 70)
