"""FastAPI application for RetailPulse AI demand forecasting & inventory optimization."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status

from src.api.schemas import (
    HealthResponse,
    DailyForecastItem,
    ForecastResponse,
    InventoryRiskItem,
    InventoryRisksResponse,
    ReplenishmentRecommendationResponse,
)
from src.database.queries import get_catalog_lists, load_sales_dataframe
from src.features.build_features import build_feature_pipeline
from src.models.multi_horizon import MULTI_HORIZON_ARTIFACT_PATH, DirectMultiHorizonForecaster
from src.optimization.inventory_logic import (
    generate_inventory_replenishment_table,
    calculate_item_inventory_policy,
)

# Global in-memory cache for models and catalog loaded at startup
MODELS_CACHE: Dict[str, Any] = {
    "multi_horizon": None,
    "stores": {},
    "products": {},
    "latest_features": None,
}


def load_application_artifacts():
    """Load trained models and catalog into memory once at application startup."""
    # 1. Load multi-horizon forecaster
    if MULTI_HORIZON_ARTIFACT_PATH.exists():
        MODELS_CACHE["multi_horizon"] = joblib.load(MULTI_HORIZON_ARTIFACT_PATH)
    
    # 2. Cache stores and products for low-latency validation
    stores, products = get_catalog_lists()
    MODELS_CACHE["stores"] = {s["store_id"]: s for s in stores}
    MODELS_CACHE["products"] = {p["product_id"]: p for p in products}

    # 3. Cache latest feature vector to avoid re-aggregating on every request
    raw_sales = load_sales_dataframe()
    MODELS_CACHE["latest_features"] = build_feature_pipeline(raw_sales, drop_na=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context: load models on startup, clean up on shutdown."""
    load_application_artifacts()
    yield
    MODELS_CACHE.clear()


app = FastAPI(
    title="RetailPulse AI Gateway",
    description="Production-grade REST API for Retail Demand Forecasting & Inventory Replenishment.",
    version="0.1.0",
    lifespan=lifespan,
)


def _validate_ids(store_id: int, product_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate existence of store and product; raise 404 if not found."""
    stores_cache = MODELS_CACHE.get("stores", {})
    products_cache = MODELS_CACHE.get("products", {})

    if store_id not in stores_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Store with ID {store_id} not found. Valid store IDs: {list(stores_cache.keys())}",
        )
    if product_id not in products_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {product_id} not found. Valid product IDs: {list(products_cache.keys())}",
        )
    return stores_cache[store_id], products_cache[product_id]


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def get_health():
    """System health check and model readiness probe."""
    is_loaded = MODELS_CACHE["multi_horizon"] is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        version="0.1.0",
        service="RetailPulse AI FastAPI Service",
        models_loaded=is_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def get_demand_forecast(
    store_id: int = Query(..., description="Store ID (e.g. 1 to 5)"),
    product_id: int = Query(..., description="Product SKU ID (e.g. 1 to 15)"),
):
    """Return next 14 days of demand forecasts with P10, P50, and P90 uncertainty bands."""
    store, product = _validate_ids(store_id, product_id)
    forecaster: DirectMultiHorizonForecaster = MODELS_CACHE["multi_horizon"]
    features_df = MODELS_CACHE["latest_features"]

    # Locate latest historical feature vector for this (store, product)
    item_feat = features_df[
        (features_df["store_id"] == store_id) & (features_df["product_id"] == product_id)
    ].sort_values("sale_date").iloc[-1:]

    if item_feat.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No historical sales history available for Store {store_id}, Product {product_id}.",
        )

    origin_date = pd.to_datetime(item_feat["sale_date"].iloc[0])
    forecast_items = []

    # Direct 14-day prediction loop
    for h in range(1, 15):
        model_h = forecaster.models[h]
        feat_cols = forecaster.feature_cols_by_horizon[h]

        target_dt = origin_date + timedelta(days=h)
        X_h = item_feat.copy()
        X_h["target_day_of_week"] = target_dt.day_of_week
        X_h["target_is_weekend"] = int(target_dt.day_of_week in [5, 6])
        X_h["target_month"] = target_dt.month
        X_h["target_on_promotion"] = 0

        p50 = float(np.clip(model_h.predict(X_h[feat_cols])[0], 0, None))
        p10 = float(np.clip(p50 * 0.75, 0, None))  # Lower demand bound
        p90 = float(np.clip(p50 * 1.35, 0, None))  # Upper peak bound

        forecast_items.append(
            DailyForecastItem(
                horizon_day=h,
                target_date=target_dt.strftime("%Y-%m-%d"),
                day_of_week=target_dt.strftime("%A"),
                predicted_p10=round(p10, 2),
                predicted_p50=round(p50, 2),
                predicted_p90=round(p90, 2),
            )
        )

    return ForecastResponse(
        store_id=store_id,
        store_name=store["store_name"],
        product_id=product_id,
        sku=product["sku"],
        product_name=product["product_name"],
        category=product["category"],
        forecast_horizon_days=14,
        forecasts=forecast_items,
    )


@app.get("/inventory/risks", response_model=InventoryRisksResponse, tags=["Inventory"])
def get_inventory_risks(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Optional filter by status: CRITICAL, WARNING, or HEALTHY",
    )
):
    """Return all catalog inventory items enriched with stockout status and replenishment recommendations."""
    df_rep = generate_inventory_replenishment_table()

    critical_count = int((df_rep["status"] == "CRITICAL").sum())
    warning_count = int((df_rep["status"] == "WARNING").sum())
    healthy_count = int((df_rep["status"] == "HEALTHY").sum())

    if status_filter:
        valid_statuses = {"CRITICAL", "WARNING", "HEALTHY"}
        filter_upper = status_filter.upper()
        if filter_upper not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{status_filter}'. Allowed: {list(valid_statuses)}",
            )
        df_rep = df_rep[df_rep["status"] == filter_upper]

    items = [InventoryRiskItem(**row) for row in df_rep.to_dict(orient="records")]

    return InventoryRisksResponse(
        total_items=len(items),
        critical_count=critical_count,
        warning_count=warning_count,
        healthy_count=healthy_count,
        items=items,
    )


@app.get(
    "/replenishment/recommend",
    response_model=ReplenishmentRecommendationResponse,
    tags=["Replenishment"],
)
def get_replenishment_recommendation(
    store_id: int = Query(..., description="Store ID (e.g. 1 to 5)"),
    product_id: int = Query(..., description="Product SKU ID (e.g. 1 to 15)"),
):
    """Return specific replenishment purchase order recommendation for an individual SKU at a store."""
    store, product = _validate_ids(store_id, product_id)
    df_rep = generate_inventory_replenishment_table()

    match = df_rep[(df_rep["store_id"] == store_id) & (df_rep["product_id"] == product_id)]
    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No replenishment profile found for Store {store_id}, Product {product_id}.",
        )

    row = match.iloc[0]
    action_required = row["recommended_order_quantity"] > 0
    if row["status"] == "CRITICAL":
        note = f"URGENT: Current stock ({row['current_stock']}) is below Safety Stock ({row['safety_stock']}). Reorder immediately."
    elif row["status"] == "WARNING":
        note = f"ATTENTION: Reorder point ({row['reorder_point']}) reached. Place replenishment order of {row['recommended_order_quantity']} units."
    else:
        note = f"HEALTHY: Current stock ({row['current_stock']}) is above reorder point. No reorder required."

    return ReplenishmentRecommendationResponse(
        store_id=store_id,
        store_name=row["store_name"],
        product_id=product_id,
        sku=row["sku"],
        product_name=row["product_name"],
        category=row["category"],
        current_stock=int(row["current_stock"]),
        safety_stock=int(row["safety_stock"]),
        reorder_point=int(row["reorder_point"]),
        status=row["status"],
        target_stock=int(row["target_stock"]),
        recommended_order_quantity=int(row["recommended_order_quantity"]),
        unit_cost=float(row["unit_cost"]),
        estimated_order_cost=float(row["estimated_order_cost"]),
        action_required=action_required,
        recommendation_note=note,
    )
