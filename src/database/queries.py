"""Reusable SQL query helpers for RetailPulse AI analytical workloads."""

import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.connection import SessionLocal
from src.database.models import Store, Product, InventoryLevel, SalesTransaction


def load_sales_dataframe(
    store_id: Optional[int] = None,
    product_id: Optional[int] = None,
) -> pd.DataFrame:
    """Load historical sales transactions as a pandas DataFrame for ML/Analytics."""
    db: Session = SessionLocal()
    try:
        query = db.query(
            SalesTransaction.transaction_id,
            SalesTransaction.sale_date,
            SalesTransaction.store_id,
            SalesTransaction.product_id,
            SalesTransaction.quantity_sold,
            SalesTransaction.revenue,
            SalesTransaction.on_promotion,
            Product.sku,
            Product.product_name,
            Product.category,
            Product.unit_cost,
            Product.unit_price,
            Product.lead_time_days,
            Store.store_name,
            Store.city,
            Store.store_type,
        ).join(Product, SalesTransaction.product_id == Product.product_id)\
         .join(Store, SalesTransaction.store_id == Store.store_id)

        if store_id:
            query = query.filter(SalesTransaction.store_id == store_id)
        if product_id:
            query = query.filter(SalesTransaction.product_id == product_id)

        df = pd.read_sql(query.statement, db.bind)
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        return df.sort_values(by=["store_id", "product_id", "sale_date"]).reset_index(drop=True)
    finally:
        db.close()


def get_inventory_status_summary() -> List[Dict[str, Any]]:
    """Retrieve current inventory snapshot enriched with risk severity status."""
    db: Session = SessionLocal()
    try:
        query = db.query(
            InventoryLevel.store_id,
            Store.store_name,
            InventoryLevel.product_id,
            Product.sku,
            Product.product_name,
            Product.category,
            Product.unit_cost,
            Product.unit_price,
            Product.lead_time_days,
            InventoryLevel.current_stock,
            InventoryLevel.safety_stock,
            InventoryLevel.reorder_point,
            InventoryLevel.is_simulated,
        ).join(Product, InventoryLevel.product_id == Product.product_id)\
         .join(Store, InventoryLevel.store_id == Store.store_id)

        rows = query.all()
        results = []
        for r in rows:
            # Risk policy classification
            if r.current_stock < r.safety_stock:
                status = "CRITICAL"
                urgency = 1
            elif r.current_stock <= r.reorder_point:
                status = "WARNING"
                urgency = 2
            else:
                status = "HEALTHY"
                urgency = 3

            results.append({
                "store_id": r.store_id,
                "store_name": r.store_name,
                "product_id": r.product_id,
                "sku": r.sku,
                "product_name": r.product_name,
                "category": r.category,
                "unit_cost": float(r.unit_cost),
                "unit_price": float(r.unit_price),
                "lead_time_days": r.lead_time_days,
                "current_stock": r.current_stock,
                "safety_stock": r.safety_stock,
                "reorder_point": r.reorder_point,
                "status": status,
                "urgency": urgency,
                "is_simulated": r.is_simulated,
            })

        return sorted(results, key=lambda x: (x["urgency"], x["store_id"]))
    finally:
        db.close()


def get_catalog_lists():
    """Retrieve store and product lists for dashboard selectors."""
    db: Session = SessionLocal()
    try:
        stores = db.query(Store).order_by(Store.store_id).all()
        products = db.query(Product).order_by(Product.product_id).all()
        return [
            {"store_id": s.store_id, "store_name": s.store_name, "city": s.city}
            for s in stores
        ], [
            {"product_id": p.product_id, "sku": p.sku, "product_name": p.product_name, "category": p.category}
            for p in products
        ]
    finally:
        db.close()
