
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Retail Analytics",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Smart Retail Analytics & Prediction")
st.caption("Sales insights • Demand forecasting • Customer churn analysis")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    # Support datasets stored either in the repository root or in ./data
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" if (base_dir / "data").is_dir() else base_dir

    sales_path = data_dir / "sales.csv"
    products_path = data_dir / "products.xlsx"
    customers_path = data_dir / "customers.csv"
    feedback_path = data_dir / "feedback.json"

    sales = pd.read_csv(sales_path, parse_dates=["date"])
    products = pd.read_excel(products_path)
    customers = pd.read_csv(customers_path)

    with open(feedback_path, "r", encoding="utf-8") as f:
        feedback = pd.DataFrame(json.load(f))

    # Normalize date column and merge sales with product details
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    retail = sales.merge(products, on="product_id", how="left")
    retail["revenue"] = retail["units_sold"] * retail["unit_price"]

    return retail, products, customers, feedback


# IMPORTANT: load_data() must be called before using these variables.
try:
    retail, products, customers, feedback = load_data()
except Exception as error:
    st.error(f"Unable to load datasets: {error}")
    st.info(
        "Confirm sales.csv, products.xlsx, customers.csv, and feedback.json "
        "are in the repository root or in its data/ folder."
    )
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

product_options = ["All Products"] + products["product_name"].tolist()
selected_product = st.sidebar.selectbox(
    "Select product",
    product_options
)

min_date = retail["date"].min().date()
max_date = retail["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = retail.copy()

if selected_product != "All Products":
    filtered = filtered[]
        filtered["product_name"] == selected_product
