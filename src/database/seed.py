"""Database seed and ingestion pipeline for RetailPulse AI.

Populates stores, product catalog, 1-year historical sales transactions,
and clearly labeled simulated operational inventory snapshots.
"""

import random
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from sqlalchemy.orm import Session

from src.database.connection import engine, SessionLocal, Base
from src.database.models import Store, Product, InventoryLevel, SalesTransaction


def create_schema():
    """Create all relational tables in the target database."""
    Base.metadata.create_all(bind=engine)


def get_curated_catalog():
    """Return realistic retail stores and product catalog with operational attributes."""
    stores_data = [
        {"store_id": 1, "store_name": "Downtown Supercenter", "city": "Chicago", "state": "IL", "store_type": "Supercenter"},
        {"store_id": 2, "store_name": "Suburban Hypermarket", "city": "Naperville", "state": "IL", "store_type": "Hypermarket"},
        {"store_id": 3, "store_name": "Metro Express", "city": "Chicago", "state": "IL", "store_type": "Express"},
        {"store_id": 4, "store_name": "Northshore Mart", "city": "Evanston", "state": "IL", "store_type": "Supermarket"},
        {"store_id": 5, "store_name": "Westside Market", "city": "Oak Park", "state": "IL", "store_type": "Supermarket"},
    ]

    products_data = [
        # Dairy
        {"product_id": 1, "sku": "SKU-DAIRY-001", "product_name": "Organic Whole Milk 1 Gal", "category": "Dairy", "unit_cost": 2.40, "unit_price": 3.99, "lead_time_days": 3},
        {"product_id": 2, "sku": "SKU-DAIRY-002", "product_name": "Greek Yogurt Plain 32oz", "category": "Dairy", "unit_cost": 3.10, "unit_price": 5.49, "lead_time_days": 4},
        {"product_id": 3, "sku": "SKU-DAIRY-003", "product_name": "Salted Butter 16oz", "category": "Dairy", "unit_cost": 2.80, "unit_price": 4.69, "lead_time_days": 5},
        {"product_id": 4, "sku": "SKU-DAIRY-004", "product_name": "Cheddar Cheese Block 8oz", "category": "Dairy", "unit_cost": 2.10, "unit_price": 3.89, "lead_time_days": 5},
        # Bakery
        {"product_id": 5, "sku": "SKU-BAKE-001", "product_name": "Artisan Sourdough Bread", "category": "Bakery", "unit_cost": 1.80, "unit_price": 4.29, "lead_time_days": 2},
        {"product_id": 6, "sku": "SKU-BAKE-002", "product_name": "Whole Wheat Sandwich Loaf", "category": "Bakery", "unit_cost": 1.40, "unit_price": 2.99, "lead_time_days": 2},
        {"product_id": 7, "sku": "SKU-BAKE-003", "product_name": "Butter Croissant 4-Pack", "category": "Bakery", "unit_cost": 2.20, "unit_price": 4.99, "lead_time_days": 3},
        # Produce
        {"product_id": 8, "sku": "SKU-PROD-001", "product_name": "Hass Avocados Bag", "category": "Produce", "unit_cost": 2.50, "unit_price": 4.99, "lead_time_days": 4},
        {"product_id": 9, "sku": "SKU-PROD-002", "product_name": "Organic Bananas Bunch", "category": "Produce", "unit_cost": 0.90, "unit_price": 1.89, "lead_time_days": 3},
        {"product_id": 10, "sku": "SKU-PROD-003", "product_name": "Honeycrisp Apples 3lb", "category": "Produce", "unit_cost": 3.20, "unit_price": 5.99, "lead_time_days": 5},
        {"product_id": 11, "sku": "SKU-PROD-004", "product_name": "Baby Spinach 16oz", "category": "Produce", "unit_cost": 1.90, "unit_price": 3.79, "lead_time_days": 3},
        # Beverages
        {"product_id": 12, "sku": "SKU-BEV-001", "product_name": "Cold Brew Coffee 32oz", "category": "Beverages", "unit_cost": 3.40, "unit_price": 5.99, "lead_time_days": 5},
        {"product_id": 13, "sku": "SKU-BEV-002", "product_name": "Sparkling Water 12pk", "category": "Beverages", "unit_cost": 4.10, "unit_price": 6.99, "lead_time_days": 7},
        {"product_id": 14, "sku": "SKU-BEV-003", "product_name": "Pure Orange Juice 52oz", "category": "Beverages", "unit_cost": 2.60, "unit_price": 4.49, "lead_time_days": 4},
        {"product_id": 15, "sku": "SKU-BEV-004", "product_name": "Organic Green Tea 16oz", "category": "Beverages", "unit_cost": 1.50, "unit_price": 2.89, "lead_time_days": 6},
    ]

    return stores_data, products_data


def generate_sales_time_series(stores, products, start_date=date(2023, 1, 1), days=365):
    """Generate realistic daily sales time-series with weekend spikes and promotions."""
    np.random.seed(42)
    random.seed(42)

    store_multipliers = {1: 1.35, 2: 1.25, 3: 0.70, 4: 1.00, 5: 0.95}
    base_demands = {
        1: 45, 2: 25, 3: 20, 4: 18,
        5: 35, 6: 40, 7: 22,
        8: 30, 9: 60, 10: 25, 11: 20,
        12: 28, 13: 35, 14: 30, 15: 15,
    }

    records = []
    product_price_map = {p["product_id"]: p["unit_price"] for p in products}

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        dow = current_date.weekday()  # Monday=0, Sunday=6
        
        # Day-of-week factor: Weekend peak (Fri/Sat/Sun)
        dow_factor = 1.0
        if dow in [4, 5]:  # Fri, Sat
            dow_factor = 1.40
        elif dow == 6:     # Sun
            dow_factor = 1.25
        elif dow == 0:     # Mon
            dow_factor = 0.85

        # Monthly seasonality (Summer vs Winter)
        month = current_date.month
        summer_factor = 1.15 if month in [6, 7, 8] else 1.0

        for store in stores:
            s_id = store["store_id"]
            s_mult = store_multipliers[s_id]

            for product in products:
                p_id = product["product_id"]
                base_d = base_demands[p_id]

                # Bi-weekly promotional events (approx 12% probability)
                is_promo = random.random() < 0.12
                promo_boost = 1.55 if is_promo else 1.0

                expected_demand = base_d * s_mult * dow_factor * summer_factor * promo_boost
                
                # Sample daily units sold from Poisson distribution
                qty = int(np.random.poisson(lam=max(1.0, expected_demand)))
                price = product_price_map[p_id]
                revenue = round(qty * price, 2)

                records.append({
                    "sale_date": current_date,
                    "store_id": s_id,
                    "product_id": p_id,
                    "quantity_sold": qty,
                    "revenue": revenue,
                    "on_promotion": is_promo,
                })

    return records


def calculate_simulated_inventory(sales_records, products, stores):
    """Calculate realistic baseline inventory, safety stock, and reorder points."""
    df_sales = pd.DataFrame(sales_records)
    
    # Calculate empirical daily mean and standard deviation per (store_id, product_id)
    stats = df_sales.groupby(["store_id", "product_id"])["quantity_sold"].agg(["mean", "std"]).reset_index()

    product_lead_times = {p["product_id"]: p["lead_time_days"] for p in products}

    inventory_records = []
    # Seed for deterministic simulation
    random.seed(101)

    for _, row in stats.iterrows():
        s_id = int(row["store_id"])
        p_id = int(row["product_id"])
        mean_d = float(row["mean"])
        std_d = float(row["std"])
        lead_time = product_lead_times[p_id]

        # Service level factor: Z=1.65 (95% service level)
        safety_stock = int(np.ceil(1.65 * std_d * np.sqrt(lead_time)))
        reorder_point = int(np.ceil(mean_d * lead_time + safety_stock))

        # Intentionally create varied realistic stock scenarios:
        # - 70% Healthy: Stock well above ROP
        # - 20% Warning: Stock near or at ROP
        # - 10% Critical: Stock below Safety Stock (urgent replenishment)
        dice = random.random()
        if dice < 0.10:
            current_stock = max(2, int(safety_stock * random.uniform(0.2, 0.8)))  # Critical Stockout Risk
        elif dice < 0.30:
            current_stock = int(reorder_point * random.uniform(0.9, 1.05))        # Warning Reorder Trigger
        else:
            current_stock = int(reorder_point * random.uniform(1.2, 2.2))         # Healthy

        inventory_records.append({
            "store_id": s_id,
            "product_id": p_id,
            "current_stock": current_stock,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "last_updated": datetime.now(timezone.utc),
            "is_simulated": True,
        })

    return inventory_records


def seed_database():
    """Execute complete schema creation and data population pipeline."""
    create_schema()
    stores_data, products_data = get_curated_catalog()

    db: Session = SessionLocal()
    try:
        # 1. Clear existing data to maintain idempotent seeding
        db.query(SalesTransaction).delete()
        db.query(InventoryLevel).delete()
        db.query(Product).delete()
        db.query(Store).delete()
        db.commit()

        # 2. Populate Stores & Products
        stores_objs = [Store(**s) for s in stores_data]
        products_objs = [Product(**p, is_simulated_metadata=True) for p in products_data]
        db.add_all(stores_objs)
        db.add_all(products_objs)
        db.commit()

        # 3. Generate and insert Sales Transactions
        sales_records = generate_sales_time_series(stores_data, products_data)
        db.bulk_insert_mappings(SalesTransaction, sales_records)
        db.commit()

        # 4. Generate and insert Simulated Inventory Snapshots
        inventory_records = calculate_simulated_inventory(sales_records, products_data, stores_data)
        db.bulk_insert_mappings(InventoryLevel, inventory_records)
        db.commit()

        # 5. Export a local raw backup CSV for auditability
        raw_csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "sales_transactions_sample.csv"
        pd.DataFrame(sales_records).to_csv(raw_csv_path, index=False)

        print(f"[SUCCESS] Seeding Complete:")
        print(f"   - Stores: {len(stores_objs)}")
        print(f"   - Products: {len(products_objs)}")
        print(f"   - Sales Transactions: {len(sales_records):,} records (365 days)")
        print(f"   - Inventory Snapshots: {len(inventory_records)} (clearly labeled is_simulated=True)")
        print(f"   - Backup CSV saved: {raw_csv_path}")

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
