"""Integration tests for RetailPulse AI FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with FastAPI lifespan triggered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Test system health check and model loading diagnostic."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert "version" in data
    assert "timestamp" in data


def test_forecast_endpoint_valid_item(client: TestClient):
    """Test 14-day multi-horizon demand forecast for valid store and product."""
    response = client.get("/forecast?store_id=1&product_id=1")
    assert response.status_code == 200
    data = response.json()

    assert data["store_id"] == 1
    assert data["product_id"] == 1
    assert data["forecast_horizon_days"] == 14
    assert len(data["forecasts"]) == 14

    for day_forecast in data["forecasts"]:
        assert 1 <= day_forecast["horizon_day"] <= 14
        assert "target_date" in day_forecast
        assert "day_of_week" in day_forecast
        p10 = day_forecast["predicted_p10"]
        p50 = day_forecast["predicted_p50"]
        p90 = day_forecast["predicted_p90"]
        assert p10 >= 0
        assert p10 <= p50 <= p90


def test_forecast_endpoint_invalid_store(client: TestClient):
    """Test that querying a non-existent store returns HTTP 404 with clear message."""
    response = client.get("/forecast?store_id=999&product_id=1")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Store with ID 999 not found" in detail


def test_forecast_endpoint_invalid_product(client: TestClient):
    """Test that querying a non-existent product returns HTTP 404 with clear message."""
    response = client.get("/forecast?store_id=1&product_id=999")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Product with ID 999 not found" in detail


def test_inventory_risks_all(client: TestClient):
    """Test inventory risks endpoint returns all catalog items and accurate counts."""
    response = client.get("/inventory/risks")
    assert response.status_code == 200
    data = response.json()

    assert data["total_items"] == 75  # 5 stores * 15 products
    assert data["critical_count"] + data["warning_count"] + data["healthy_count"] == 75
    assert len(data["items"]) == 75

    # Check structure of the first item
    first = data["items"][0]
    assert "sku" in first
    assert "reorder_point" in first
    assert "status" in first
    assert first["status"] in {"CRITICAL", "WARNING", "HEALTHY"}


def test_inventory_risks_filter_critical(client: TestClient):
    """Test status filter returns only matching items."""
    response = client.get("/inventory/risks?status=CRITICAL")
    assert response.status_code == 200
    data = response.json()

    assert data["total_items"] == data["critical_count"]
    for item in data["items"]:
        assert item["status"] == "CRITICAL"


def test_inventory_risks_invalid_filter(client: TestClient):
    """Test invalid status filter returns HTTP 400 Bad Request."""
    response = client.get("/inventory/risks?status=UNKNOWN_STATUS")
    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]


def test_replenishment_recommendation_valid(client: TestClient):
    """Test purchase order recommendation endpoint for a specific SKU-store pair."""
    response = client.get("/replenishment/recommend?store_id=1&product_id=1")
    assert response.status_code == 200
    data = response.json()

    assert data["store_id"] == 1
    assert data["product_id"] == 1
    assert "sku" in data
    assert "reorder_point" in data
    assert "target_stock" in data
    assert "recommended_order_quantity" in data
    assert "action_required" in data
    assert "recommendation_note" in data
    assert isinstance(data["action_required"], bool)


def test_replenishment_recommendation_invalid(client: TestClient):
    """Test replenishment recommendation endpoint for an invalid store ID returns 404."""
    response = client.get("/replenishment/recommend?store_id=888&product_id=1")
    assert response.status_code == 404
    assert "Store with ID 888 not found" in response.json()["detail"]
