"""Pydantic request and response schemas for RetailPulse AI REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System and model service health diagnostic."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "0.1.0"})
    service: str = Field(..., json_schema_extra={"example": "RetailPulse AI FastAPI Service"})
    models_loaded: bool = Field(..., json_schema_extra={"example": True})
    timestamp: str


class DailyForecastItem(BaseModel):
    """Single day forecast item across the 14-day horizon."""
    horizon_day: int = Field(..., ge=1, le=14, json_schema_extra={"example": 1})
    target_date: str = Field(..., json_schema_extra={"example": "2024-01-01"})
    day_of_week: str = Field(..., json_schema_extra={"example": "Monday"})
    predicted_p10: float = Field(..., description="10th percentile pessimistic demand floor")
    predicted_p50: float = Field(..., description="50th percentile median expected demand")
    predicted_p90: float = Field(..., description="90th percentile peak demand ceiling")


class ForecastResponse(BaseModel):
    """Multi-horizon 14-day demand forecast payload."""
    store_id: int = Field(..., json_schema_extra={"example": 1})
    store_name: str = Field(..., json_schema_extra={"example": "Downtown Supercenter"})
    product_id: int = Field(..., json_schema_extra={"example": 1})
    sku: str = Field(..., json_schema_extra={"example": "SKU-DAIRY-001"})
    product_name: str = Field(..., json_schema_extra={"example": "Organic Whole Milk 1 Gal"})
    category: str = Field(..., json_schema_extra={"example": "Dairy"})
    forecast_horizon_days: int = Field(..., json_schema_extra={"example": 14})
    forecasts: List[DailyForecastItem]


class InventoryRiskItem(BaseModel):
    """Inventory status and replenishment recommendation for an individual item."""
    store_id: int
    store_name: str
    product_id: int
    sku: str
    product_name: str
    category: str
    lead_time_days: int
    current_stock: int
    in_transit: int
    safety_stock: int
    reorder_point: int
    status: str = Field(..., description="CRITICAL, WARNING, or HEALTHY")
    urgency: int
    target_stock: int
    recommended_order_quantity: int
    estimated_order_cost: float


class InventoryRisksResponse(BaseModel):
    """Complete catalog inventory risk matrix with summary counts."""
    total_items: int
    critical_count: int
    warning_count: int
    healthy_count: int
    items: List[InventoryRiskItem]


class ReplenishmentRecommendationResponse(BaseModel):
    """Direct replenishment order recommendation for a specific SKU-store pair."""
    store_id: int
    store_name: str
    product_id: int
    sku: str
    product_name: str
    category: str
    current_stock: int
    safety_stock: int
    reorder_point: int
    status: str
    target_stock: int
    recommended_order_quantity: int
    unit_cost: float
    estimated_order_cost: float
    action_required: bool
    recommendation_note: str

