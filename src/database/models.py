"""SQLAlchemy ORM models for RetailPulse AI.

Explicitly differentiates empirical sales transactions from simulated operational data.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Numeric,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship
from src.database.connection import Base


class Store(Base):
    """Retail store entity."""

    __tablename__ = "stores"

    store_id = Column(Integer, primary_key=True, index=True)
    store_name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=False)
    store_type = Column(String(30), nullable=False)  # Supermarket, Express, Flagship

    sales = relationship("SalesTransaction", back_populates="store", cascade="all, delete-orphan")
    inventory = relationship("InventoryLevel", back_populates="store", cascade="all, delete-orphan")


class Product(Base):
    """Product catalog with SKU-level attributes."""

    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(50), unique=True, nullable=False, index=True)
    product_name = Column(String(150), nullable=False)
    category = Column(String(50), nullable=False, index=True)
    
    # Financial and supply chain attributes
    unit_cost = Column(Numeric(10, 2), nullable=False)
    unit_price = Column(Numeric(10, 2), nullable=False)
    lead_time_days = Column(Integer, nullable=False, default=7)  # Operational supplier lead time
    
    # Transparency flag: Explicitly denotes simulated supply-chain attributes
    is_simulated_metadata = Column(Boolean, default=True, nullable=False)

    sales = relationship("SalesTransaction", back_populates="product", cascade="all, delete-orphan")
    inventory = relationship("InventoryLevel", back_populates="product", cascade="all, delete-orphan")


class InventoryLevel(Base):
    """Current warehouse / store stock level snapshot."""

    __tablename__ = "inventory_levels"

    inventory_id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False, index=True)
    
    current_stock = Column(Integer, nullable=False, default=0)
    safety_stock = Column(Integer, nullable=False, default=0)
    reorder_point = Column(Integer, nullable=False, default=0)
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Transparency flag: Explicitly marks simulated inventory levels
    is_simulated = Column(Boolean, default=True, nullable=False)

    store = relationship("Store", back_populates="inventory")
    product = relationship("Product", back_populates="inventory")

    __table_args__ = (
        Index("idx_store_product_inventory", "store_id", "product_id", unique=True),
    )


class SalesTransaction(Base):
    """Historical daily sales demand observations."""

    __tablename__ = "sales_transactions"

    transaction_id = Column(Integer, primary_key=True, autoincrement=True)
    sale_date = Column(Date, nullable=False, index=True)
    store_id = Column(Integer, ForeignKey("stores.store_id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False, index=True)
    
    quantity_sold = Column(Integer, nullable=False, default=0)
    revenue = Column(Numeric(12, 2), nullable=False, default=0.00)
    on_promotion = Column(Boolean, default=False, nullable=False)

    store = relationship("Store", back_populates="sales")
    product = relationship("Product", back_populates="sales")

    __table_args__ = (
        Index("idx_sales_store_product_date", "store_id", "product_id", "sale_date"),
    )
