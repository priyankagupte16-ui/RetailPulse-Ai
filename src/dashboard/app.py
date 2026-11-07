"""Streamlit dashboard for RetailPulse AI.

This dashboard does **no** business logic itself – it calls the FastAPI
backend (running at http://localhost:8000 by default) and visualises the
responses.

Four pages are provided via a sidebar selector:

1. **Overview** – KPI cards for the number of items in each risk
   category and a bar‑chart of risk counts per product category.
2. **Forecast Explorer** – Choose a store and product; a Plotly line chart
   displays the 14‑day forecast (P10, P50, P90) with a shaded confidence
   band.
3. **Reorder Table** – Full inventory‑risk table, colour‑coded by status,
   optional status filter and a *Download CSV* button.
4. **What‑If Simulator** – Sliders let you change the service‑level target
   (affects the safety‑stock multiplier) and an additional lead‑time delay.
   The table updates in‑place to show the new safety stock and reorder
   point.

The code follows a clean, simple style with a single colour theme.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_ROOT = "http://localhost:8000"  # FastAPI base URL
THEME_COLOR = "#2C7BB6"  # Consistent blue theme for charts / metrics

# ---------------------------------------------------------------------------
# Helper functions – all data is fetched from the FastAPI service.
# ---------------------------------------------------------------------------
@st.experimental_singleton
def get_inventory_data() -> pd.DataFrame:
    """Fetch the complete inventory‑risk matrix.

    Returns a pandas DataFrame with one row per (store, product) item.
    """
    resp = httpx.get(f"{API_ROOT}/inventory/risks")
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data["items"])

@st.experimental_singleton
def get_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Load the store and product catalogs used for dropdowns.

    The FastAPI service provides them via the same inventory endpoint –
    we extract the unique values.
    """
    df = get_inventory_data()
    stores = (
        df[["store_id", "store_name"]]
        .drop_duplicates()
        .sort_values("store_id")
        .to_dict(orient="records")
    )
    products = (
        df[["product_id", "sku", "product_name", "category"]]
        .drop_duplicates()
        .sort_values("product_id")
        .to_dict(orient="records")
    )
    return {"stores": stores, "products": products}

def fetch_forecast(store_id: int, product_id: int) -> pd.DataFrame:
    """Call the /forecast endpoint and return a tidy DataFrame.
    """
    resp = httpx.get(
        f"{API_ROOT}/forecast",
        params={"store_id": store_id, "product_id": product_id},
    )
    resp.raise_for_status()
    payload = resp.json()
    forecasts = payload["forecasts"]
    return pd.DataFrame(forecasts)

# ---------------------------------------------------------------------------
# Page implementations
# ---------------------------------------------------------------------------
def page_overview() -> None:
    """Show KPI cards and a risk‑by‑category bar chart."""
    st.title("📊 Overview")
    df = get_inventory_data()

    # KPI cards – total counts per status
    status_counts = df["status"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("CRITICAL", int(status_counts.get("CRITICAL", 0)))
    col2.metric("WARNING", int(status_counts.get("WARNING", 0)))
    col3.metric("HEALTHY", int(status_counts.get("HEALTHY", 0)))

    # Bar chart – risk count per product category
    cat_counts = (
        df.groupby("category")["status"]
        .count()
        .reset_index(name="items")
        .sort_values("items", ascending=False)
    )
    fig = px.bar(
        cat_counts,
        x="category",
        y="items",
        color="category",
        color_discrete_sequence=[THEME_COLOR],
        title="Items by Category",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_forecast_explorer() -> None:
    """Allow the user to pick a store/product and visualise the 14‑day forecast.
    """
    st.title("🔮 Forecast Explorer")
    catalog = get_catalog()
    store_options = {s["store_name"]: s["store_id"] for s in catalog["stores"]}
    product_options = {
        f"{p['sku']} – {p['product_name']}": p["product_id"]
        for p in catalog["products"]
    }

    store_name = st.selectbox("Store", options=list(store_options.keys()))
    product_name = st.selectbox("Product", options=list(product_options.keys()))
    store_id = store_options[store_name]
    product_id = product_options[product_name]

    try:
        df = fetch_forecast(store_id, product_id)
    except httpx.HTTPStatusError as exc:
        st.error(f"Failed to load forecast: {exc.response.status_code} – {exc.response.text}")
        return

    # Plot: p50 line plus shaded p10‑p90 band
    fig = px.line(
        df,
        x="target_date",
        y="predicted_p50",
        title=f"14‑day Forecast for {product_name} (Store {store_name})",
        markers=True,
    )
    fig.add_trace(
        px.scatter(
            df,
            x="target_date",
            y="predicted_p10",
            opacity=0,
        ).data[0]
    )
    fig.add_trace(
        px.scatter(
            df,
            x="target_date",
            y="predicted_p90",
            opacity=0,
        ).data[0]
    )
    fig.update_traces(showlegend=False)
    fig.add_traces(
        px.
        area(
            df,
            x="target_date",
            y=["predicted_p10", "predicted_p90"],
            color_discrete_sequence=["#d1e5f0", "#d1e5f0"],
        ).data
    )
    # The area plot adds two traces (p10 and p90); we keep only the filled band.
    # Hide the individual lines – they are already covered by the shaded area.
    fig.update_traces(selector=dict(type="scatter", mode="lines"), line=dict(width=0))
    fig.update_layout(yaxis_title="Units", xaxis_title="Date", legend_title="")
    st.plotly_chart(fig, use_container_width=True)


def page_reorder_table() -> None:
    """Display the inventory‑risk table with colour coding and CSV download."""
    st.title("📦 Reorder Table")
    df = get_inventory_data()

    # Optional status filter
    status_filter = st.selectbox(
        "Filter by status",
        options=["All", "CRITICAL", "WARNING", "HEALTHY"],
        index=0,
    )
    if status_filter != "All":
        df = df[df["status"] == status_filter]

    # Colour‑code rows – we use st.dataframe styling.
    def colour_rows(row: pd.Series) -> List[str]:
        if row["status"] == "CRITICAL":
            return ["background-color: #ffcccc;"] * len(row)
        if row["status"] == "WARNING":
            return ["background-color: #fff4cc;"] * len(row)
        return ["background-color: #ccffcc;"] * len(row)

    st.dataframe(df.style.apply(colour_rows, axis=1), use_container_width=True)

    # CSV download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="inventory_risk_table.csv",
        mime="text/csv",
    )


def page_what_if_simulator() -> None:
    """Simple what‑if tool that tweaks service level and lead‑time.

    The safety‑stock formula from ``src/optimization/inventory_logic`` is:
        safety_stock = max(1, sum(P90[:lead]) - sum(P50[:lead]))
    We approximate the effect of a higher service level by scaling the
    P90‑to‑P50 gap with a multiplier derived from the slider (e.g. 0.9 →
    multiplier 1.0, 0.99 → 1.2). The lead‑time delay simply adds extra days
    to the existing lead time.
    """
    st.title("🔧 What‑If Simulator")
    df = get_inventory_data()

    # Sliders – placed in the sidebar for a compact layout
    st.sidebar.subheader("Simulation parameters")
    service_level = st.sidebar.slider(
        "Target service level (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="Higher service level means a larger safety‑stock buffer.",
    )
    lead_time_delay = st.sidebar.slider(
        "Lead‑time delay (days)",
        min_value=0,
        max_value=5,
        value=0,
        help="Additional days suppliers need to deliver after the base lead‑time.",
    )

    # Compute a simple multiplier (linear interpolation) – this is a heuristic.
    multiplier = 1.0 + (service_level - 95) * 0.02  # 95% → 1.0, 99% → 1.08

    # Re‑calculate safety stock & reorder point for each row.
    def recalc(row: pd.Series) -> pd.Series:
        lead = int(row["lead_time_days"]) + lead_time_delay
        # Use the existing P50 & P90 values that were used to produce the
        # original safety stock. In the inventory logic they are derived from
        # the 14‑day forecast – we approximate by re‑using the original
        # safety_stock as a proxy for the gap.
        # New safety stock = original gap * multiplier (rounded, min 1)
        original_gap = row["safety_stock"]
        new_gap = max(1, int(round(original_gap * multiplier)))
        new_safety = new_gap
        new_rop = row["lead_time_demand"] + new_safety
        row["lead_time_days"] = lead
        row["safety_stock"] = new_safety
        row["reorder_point"] = new_rop
        # Updated recommended order quantity based on the new ROP.
        effective_stock = row["current_stock"] + row["in_transit"]
        target_stock = new_rop + row["target_stock"] - row["reorder_point"]  # keep original 7‑day buffer
        row["recommended_order_quantity"] = max(0, target_stock - effective_stock)
        # Update status based on new numbers
        if effective_stock < new_safety:
            row["status"] = "CRITICAL"
        elif effective_stock <= new_rop:
            row["status"] = "WARNING"
        else:
            row["status"] = "HEALTHY"
        return row

    simulated_df = df.apply(recalc, axis=1)

    # Show colour‑coded table (reuse helper from previous page)
    def colour_rows(row: pd.Series) -> List[str]:
        if row["status"] == "CRITICAL":
            return ["background-color: #ffcccc;"] * len(row)
        if row["status"] == "WARNING":
            return ["background-color: #fff4cc;"] * len(row)
        return ["background-color: #ccffcc;"] * len(row)

    st.subheader("Simulated inventory policy")
    st.dataframe(simulated_df.style.apply(colour_rows, axis=1), use_container_width=True)

# ---------------------------------------------------------------------------
# Main app – sidebar navigation
# ---------------------------------------------------------------------------
PAGE_FUNCS = {
    "Overview": page_overview,
    "Forecast Explorer": page_forecast_explorer,
    "Reorder Table": page_reorder_table,
    "What‑If Simulator": page_what_if_simulator,
}

st.sidebar.title("RetailPulse Dashboard")
selected_page = st.sidebar.radio("Navigate", list(PAGE_FUNCS.keys()))
PAGE_FUNCS[selected_page]()
