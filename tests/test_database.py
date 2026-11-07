"""Unit and integration tests for database models, seeding, and queries."""

import pytest
import pandas as pd
from src.database.connection import SessionLocal
from src.database.models import Store, Product, InventoryLevel, SalesTransaction
from src.database.queries import load_sales_dataframe, get_inventory_status_summary, get_catalog_lists


@pytest.fixture(scope="module")
def db_session():
    """Provide a database session for testing."""
    session = SessionLocal()
    yield session
    session.close()


def test_catalog_counts(db_session):
    """Verify exact count of stores and products in the manageable subset."""
    store_count = db_session.query(Store).count()
    product_count = db_session.query(Product).count()

    assert store_count == 5, f"Expected 5 stores, got {store_count}"
    assert product_count == 15, f"Expected 15 products, got {product_count}"


def test_sales_transactions_volume(db_session):
    """Verify sales transactions cover full 365 days across all 5 stores and 15 products."""
    total_sales = db_session.query(SalesTransaction).count()
    expected_records = 5 * 15 * 365
    assert total_sales == expected_records, f"Expected {expected_records} sales records, got {total_sales}"


def test_simulated_data_transparency_labels(db_session):
    """Ensure simulated operational data is explicitly labeled for interview integrity."""
    unlabeled_products = db_session.query(Product).filter(Product.is_simulated_metadata.is_(False)).count()
    unlabeled_inventory = db_session.query(InventoryLevel).filter(InventoryLevel.is_simulated.is_(False)).count()

    assert unlabeled_products == 0, "All product supply chain attributes should be marked as simulated"
    assert unlabeled_inventory == 0, "All inventory snapshots must be explicitly marked is_simulated=True"


def test_load_sales_dataframe_contract():
    """Verify dataframe loader returns valid types, non-empty, and parsed timestamps."""
    df = load_sales_dataframe(store_id=1, product_id=1)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 365
    assert pd.api.types.is_datetime64_any_dtype(df["sale_date"])
    assert "quantity_sold" in df.columns
    assert "revenue" in df.columns
    assert (df["quantity_sold"] >= 0).all()


def test_inventory_risk_classification():
    """Verify risk logic categorizes inventory into CRITICAL, WARNING, and HEALTHY."""
    inv_summary = get_inventory_status_summary()
    assert len(inv_summary) == 75

    statuses = {item["status"] for item in inv_summary}
    assert "CRITICAL" in statuses, "Should detect critical stockout risks"
    assert "HEALTHY" in statuses, "Should contain healthy stock levels"

    for item in inv_summary:
        if item["current_stock"] < item["safety_stock"]:
            assert item["status"] == "CRITICAL"
        elif item["current_stock"] <= item["reorder_point"]:
            assert item["status"] == "WARNING"
        else:
            assert item["status"] == "HEALTHY"
