"""Unit tests for inventory optimization logic with 3 hand-calculated examples."""

import pytest
import numpy as np
import pandas as pd
from src.optimization.inventory_logic import (
    calculate_item_inventory_policy,
    generate_inventory_replenishment_table,
)


def test_hand_calculated_example_1_critical_stockout():
    """HAND-CALCULATED EXAMPLE 1: CRITICAL STOCKOUT RISK.

    Inputs:
    - Lead Time = 3 days
    - P50 Forecast (Days 1 to 10) = 50 units/day
    - P90 Forecast (Days 1 to 10) = 65 units/day
    - Current Stock = 20 units
    - In-Transit = 0 units

    Manual Calculations:
    - Lead Time Demand (LTD) = 50 * 3 = 150 units
    - P90 Lead Time Demand = 65 * 3 = 195 units
    - Safety Stock (SS) = 195 - 150 = 45 units
    - Reorder Point (ROP) = 150 + 45 = 195 units
    - Effective Stock = 20 + 0 = 20 units
    - Status: 20 < 45 (SS) -> CRITICAL
    - 7-Day P50 Demand = 50 * 7 = 350 units
    - Target Stock = 195 (ROP) + 350 = 545 units
    - Recommended Order Qty = 545 - 20 = 525 units
    """
    p50 = [50.0] * 10
    p90 = [65.0] * 10

    policy = calculate_item_inventory_policy(
        lead_time_days=3,
        p50_daily_forecast=p50,
        p90_daily_forecast=p90,
        current_stock=20,
        in_transit=0,
    )

    assert policy["lead_time_demand"] == 150
    assert policy["safety_stock"] == 45
    assert policy["reorder_point"] == 195
    assert policy["effective_stock"] == 20
    assert policy["status"] == "CRITICAL"
    assert policy["urgency"] == 1
    assert policy["target_stock"] == 545
    assert policy["recommended_order_quantity"] == 525


def test_hand_calculated_example_2_warning_reorder():
    """HAND-CALCULATED EXAMPLE 2: WARNING (REORDER POINT BREACHED).

    Inputs:
    - Lead Time = 4 days
    - P50 Forecast (Days 1 to 11) = 30 units/day
    - P90 Forecast (Days 1 to 11) = 40 units/day
    - Current Stock = 120 units
    - In-Transit = 20 units (Effective Stock = 140)

    Manual Calculations:
    - Lead Time Demand (LTD) = 30 * 4 = 120 units
    - P90 Lead Time Demand = 40 * 4 = 160 units
    - Safety Stock (SS) = 160 - 120 = 40 units
    - Reorder Point (ROP) = 120 + 40 = 160 units
    - Effective Stock = 120 + 20 = 140 units
    - Status: 140 >= 40 (SS) and 140 <= 160 (ROP) -> WARNING
    - 7-Day P50 Demand = 30 * 7 = 210 units
    - Target Stock = 160 + 210 = 370 units
    - Recommended Order Qty = 370 - 140 = 230 units
    """
    p50 = [30.0] * 11
    p90 = [40.0] * 11

    policy = calculate_item_inventory_policy(
        lead_time_days=4,
        p50_daily_forecast=p50,
        p90_daily_forecast=p90,
        current_stock=120,
        in_transit=20,
    )

    assert policy["lead_time_demand"] == 120
    assert policy["safety_stock"] == 40
    assert policy["reorder_point"] == 160
    assert policy["effective_stock"] == 140
    assert policy["status"] == "WARNING"
    assert policy["urgency"] == 2
    assert policy["target_stock"] == 370
    assert policy["recommended_order_quantity"] == 230


def test_hand_calculated_example_3_healthy_no_order():
    """HAND-CALCULATED EXAMPLE 3: HEALTHY (ADEQUATE INVENTORY, ZERO ORDER).

    Inputs:
    - Lead Time = 2 days
    - P50 Forecast (Days 1 to 9) = 20 units/day
    - P90 Forecast (Days 1 to 9) = 25 units/day
    - Current Stock = 300 units
    - In-Transit = 0 units

    Manual Calculations:
    - Lead Time Demand (LTD) = 20 * 2 = 40 units
    - P90 Lead Time Demand = 25 * 2 = 50 units
    - Safety Stock (SS) = 50 - 40 = 10 units
    - Reorder Point (ROP) = 40 + 10 = 50 units
    - Effective Stock = 300 units
    - Status: 300 > 50 (ROP) -> HEALTHY
    - 7-Day P50 Demand = 20 * 7 = 140 units
    - Target Stock = 50 + 140 = 190 units
    - Recommended Order Qty = max(0, 190 - 300) = 0 units (No order needed)
    """
    p50 = [20.0] * 9
    p90 = [25.0] * 9

    policy = calculate_item_inventory_policy(
        lead_time_days=2,
        p50_daily_forecast=p50,
        p90_daily_forecast=p90,
        current_stock=300,
        in_transit=0,
    )

    assert policy["lead_time_demand"] == 40
    assert policy["safety_stock"] == 10
    assert policy["reorder_point"] == 50
    assert policy["effective_stock"] == 300
    assert policy["status"] == "HEALTHY"
    assert policy["urgency"] == 3
    assert policy["target_stock"] == 190
    assert policy["recommended_order_quantity"] == 0


def test_generate_inventory_replenishment_table_contract():
    """Verify full table generation produces all 75 items with required columns."""
    df_rep = generate_inventory_replenishment_table()

    assert isinstance(df_rep, pd.DataFrame)
    assert len(df_rep) == 75, f"Expected 75 items, got {len(df_rep)}"

    required_cols = [
        "store_id", "store_name", "product_id", "sku", "product_name",
        "category", "lead_time_days", "current_stock", "lead_time_demand",
        "safety_stock", "reorder_point", "status", "target_stock",
        "recommended_order_quantity", "estimated_order_cost"
    ]
    for col in required_cols:
        assert col in df_rep.columns, f"Missing column {col} in replenishment table!"

    # Mathematical and business sanity checks
    assert (df_rep["safety_stock"] >= 1).all()
    assert (df_rep["reorder_point"] >= df_rep["lead_time_demand"]).all()
    assert (df_rep["recommended_order_quantity"] >= 0).all()
    assert set(df_rep["status"].unique()).issubset({"CRITICAL", "WARNING", "HEALTHY"})
