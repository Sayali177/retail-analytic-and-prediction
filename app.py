
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

DATA_DIR = Path(__file__).parent / "data"

st.title("🛍️ Smart Retail Analytics & Prediction")
st.caption("Sales insights • Demand forecasting • Customer churn analysis")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"])
    products = pd.read_excel(DATA_DIR / "products.xlsx")
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    with open(DATA_DIR / "feedback.json", "r", encoding="utf-8") as f:
        feedback = pd.DataFrame(json.load(f))

    retail = sales.merge(products, on="product_id", how="left")
    retail["revenue"] = retail["units_sold"] * retail["unit_price"]

    return retail, products, customers, feedback


try:
    retail, products, customers, feedback = load_data()
except Exception as error:
    st.error(f"Unable to load datasets: {error}")
    st.info("Run `python create_data.py` first.")
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
    filtered = filtered[
        filtered["product_name"] == selected_product
    ]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["date"].dt.date.between(start_date, end_date)
    ]

if filtered.empty:
    st.warning("No sales data matches the selected filters.")
    st.stop()

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------
total_revenue = filtered["revenue"].sum()
total_units = filtered["units_sold"].sum()
average_daily_revenue = (
    filtered.groupby("date")["revenue"].sum().mean()
)
number_of_products = filtered["product_id"].nunique()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,}")
c3.metric("Avg. Daily Revenue", f"₹{average_daily_revenue:,.0f}")
c4.metric("Products Sold", number_of_products)

st.divider()

# --------------------------------------------------
# SALES TREND
# --------------------------------------------------
st.subheader("📈 Sales and Revenue Trends")

daily_sales = (
    filtered.groupby("date", as_index=False)
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
)

fig = px.line(
    daily_sales,
    x="date",
    y="revenue",
    markers=True,
    title="Daily Revenue",
    labels={"date": "Date", "revenue": "Revenue (₹)"}
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PRODUCT PERFORMANCE
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🏆 Revenue by Product")

    product_revenue = (
        filtered.groupby("product_name", as_index=False)
        ["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )

    fig = px.bar(
        product_revenue,
        x="product_name",
        y="revenue",
        title="Total Revenue per Product",
        labels={
            "product_name": "Product",
            "revenue": "Revenue (₹)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📦 Current Stock")

    stock_data = products.copy()
    stock_data["stock_status"] = np.where(
        stock_data["stock"] < 25,
        "Low Stock",
        "Available"
    )

    fig = px.bar(
        stock_data,
        x="product_name",
        y="stock",
        color="stock_status",
        title="Stock Available by Product",
        labels={"product_name": "Product", "stock": "Units"}
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# DEMAND FORECASTING
# --------------------------------------------------
st.divider()
st.subheader("🔮 Product Demand Prediction")

forecast_product = st.selectbox(
    "Choose product for demand forecasting",
    products["product_name"].tolist()
)

forecast_data = retail[
    retail["product_name"] == forecast_product
].copy()

daily_demand = (
    forecast_data.groupby("date", as_index=False)
    ["units_sold"].sum()
    .sort_values("date")
)

if len(daily_demand) >= 30:
    daily_demand["day_number"] = np.arange(len(daily_demand))
    daily_demand["day_of_week"] = daily_demand["date"].dt.dayofweek
    daily_demand["rolling_7"] = (
        daily_demand["units_sold"]
        .rolling(7, min_periods=1)
        .mean()
    )

    features = ["day_number", "day_of_week", "rolling_7"]
    X = daily_demand[features]
    y = daily_demand["units_sold"]

    split_point = max(1, int(len(daily_demand) * 0.8))
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        st.metric("Historical Test MAE", f"{mae:.2f} units")

    future_rows = []
    history = daily_demand["units_sold"].tolist()
    last_day = int(daily_demand["day_number"].iloc[-1])
    last_date = daily_demand["date"].max()

    for step in range(1, 8):
        next_date = last_date + pd.Timedelta(days=step)
        rolling_average = float(np.mean(history[-7:]))

        future_X = pd.DataFrame([{
            "day_number": last_day + step,
            "day_of_week": next_date.dayofweek,
            "rolling_7": rolling_average
        }])

        predicted = max(0, int(round(model.predict(future_X)[0])))

        future_rows.append({
            "date": next_date,
            "predicted_units": predicted
        })
        history.append(predicted)

    future_df = pd.DataFrame(future_rows)

    st.write("Predicted demand for the next 7 days:")
    st.dataframe(future_df, use_container_width=True)

    historical_plot = daily_demand[["date", "units_sold"]].copy()
    historical_plot["type"] = "Actual demand"
    historical_plot = historical_plot.rename(
        columns={"units_sold": "units"}
    )

    future_plot = future_df.rename(
        columns={"predicted_units": "units"}
    )
    future_plot["type"] = "Predicted demand"

    combined_plot = pd.concat(
        [historical_plot, future_plot],
        ignore_index=True
    )

    fig = px.line(
        combined_plot,
        x="date",
        y="units",
        color="type",
        markers=True,
        title="Historical and Forecast Demand",
        labels={"date": "Date", "units": "Units Sold"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Forecasts are estimates from synthetic sample data, "
        "not guaranteed future sales."
    )
else:
    st.warning("Not enough historical data to make a forecast.")

# --------------------------------------------------
# CUSTOMER CHURN PREDICTION
# --------------------------------------------------
st.divider()
st.subheader("👥 Customer Churn Analysis")

churn_count = int(customers["churned"].sum())
churn_rate = customers["churned"].mean() * 100

a, b = st.columns(2)
a.metric("Customers in Dataset", len(customers))
b.metric("Historical Churn Rate", f"{churn_rate:.1f}%")

features = [
    "tenure_months",
    "monthly_spend",
    "purchases_last_90d",
    "days_since_last_purchase",
    "complaints_last_6m"
]

X = customers[features]
y = customers["churned"]

if y.nunique() >= 2 and y.value_counts().min() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    churn_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    churn_model.fit(X_train, y_train)

    churn_predictions = churn_model.predict(X_test)
    accuracy = accuracy_score(y_test, churn_predictions)

    st.metric("Churn Model Test Accuracy", f"{accuracy * 100:.1f}%")

    customer_results = customers.copy()
    customer_results["churn_risk"] = (
        churn_model.predict_proba(X)[:, 1] * 100
    ).round(1)

    customer_results["risk_level"] = pd.cut(
        customer_results["churn_risk"],
        bins=[-1, 35, 65, 100],
        labels=["Low", "Medium", "High"]
    )

    st.write("Customers with the highest predicted churn risk")
    st.dataframe(
        customer_results.sort_values(
            "churn_risk", ascending=False
        )[
            [
                "customer_id",
                "days_since_last_purchase",
                "complaints_last_6m",
                "churn_risk",
                "risk_level"
            ]
        ].head(10),
        use_container_width=True
    )
else:
    st.warning(
        "The churn dataset needs examples of both churn classes "
        "to train the model."
    )

# --------------------------------------------------
# CUSTOMER FEEDBACK
# --------------------------------------------------
st.divider()
st.subheader("⭐ Customer Feedback")

if not feedback.empty:
    feedback["rating"] = pd.to_numeric(
        feedback["rating"], errors="coerce"
    )

    avg_rating = feedback["rating"].mean()
    st.metric("Average Customer Rating", f"{avg_rating:.2f} / 5")

    rating_counts = (
        feedback["rating"].value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="responses")
    )

    fig = px.bar(
        rating_counts,
        x="rating",
        y="responses",
        title="Customer Rating Distribution",
        labels={"rating": "Rating", "responses": "Number of Reviews"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Recent feedback")
    st.dataframe(
        feedback.sort_values("date", ascending=False).head(10),
        use_container_width=True
    )

# --------------------------------------------------
# DOWNLOAD UNIFIED DATA
# --------------------------------------------------


# --------------------------------------------------
# SMART INVENTORY RECOMMENDATIONS
# --------------------------------------------------


# --------------------------------------------------
# WEATHER-AWARE RETAIL INSIGHTS
# --------------------------------------------------

# --------------------------------------------------
# WEATHER-BASED PRODUCT RECOMMENDATIONS
# --------------------------------------------------
st.divider()
st.header("🌦️ Weather-Based Product Recommendations")

st.write(
    "Get product suggestions based on live weather forecasts, "
    "recent sales, and current stock."
)

weather_city = st.selectbox(
    "Choose store location",
    ["Nagpur", "Pune", "Mumbai", "Delhi"],
    key="weather_city"
)

locations = {
    "Nagpur": (21.1458, 79.0882),
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090)
}

latitude, longitude = locations[weather_city]

if st.button("Generate Weather-Based Recommendations"):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,precipitation,relative_humidity_2m"
                ),
                "daily": (
                    "precipitation_probability_max,"
                    "temperature_2m_max"
                ),
                "forecast_days": 3,
                "timezone": "auto"
            },
            timeout=10
        )
        response.raise_for_status()
        weather = response.json()

        current = weather["current"]
        daily = weather["daily"]

        temperature = float(current["temperature_2m"])
        rain_probability = max(
            float(value or 0)
            for value in daily["precipitation_probability_max"]
        )
        max_temperature = max(
            float(value)
            for value in daily["temperature_2m_max"]
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Temperature", f"{temperature:.1f} °C")
        c2.metric("Forecast Maximum", f"{max_temperature:.1f} °C")
        c3.metric("Maximum Rain Probability", f"{rain_probability:.0f}%")

        # Match product names/categories to weather-sensitive needs.
        # These are illustrative rules, not learned weather effects.
        product_rules = {
            "umbrella": {
                "rain": True,
                "heat": False,
                "message": "Rain preparedness"
            },
            "raincoat": {
                "rain": True,
                "heat": False,
                "message": "Rain preparedness"
            },
            "water": {
                "rain": False,
                "heat": True,
                "message": "Hydration during hot weather"
            },
            "iced tea": {
                "rain": False,
                "heat": True,
                "message": "Cold beverage demand may be worth reviewing"
            },
            "sunscreen": {
                "rain": False,
                "heat": True,
                "message": "Sun protection during hot weather"
            }
        }

        hot_weather = max_temperature >= 32
        rainy_weather = rain_probability >= 60

        recommendations = []

        for _, product in products.iterrows():
            name = str(product["product_name"])
            name_lower = name.lower()
            category = str(product.get("category", "")).lower()
            match = None

            # Match known product names first.
            for keyword, rule in product_rules.items():
                if keyword in name_lower:
                    match = rule
                    break

            # Also support category-based matching.
            if match is None and any(
                word in category
                for word in ["beverage", "drink"]
            ):
                if hot_weather:
                    match = {
                        "rain": False,
                        "heat": True,
                        "message": "Review beverage demand in hot weather"
                    }

            if match is None:
                continue

            weather_matches = (
                (rainy_weather and match["rain"])
                or (hot_weather and match["heat"])
            )

            if not weather_matches:
                continue

            product_sales = retail[
                retail["product_id"] == product["product_id"]
            ].sort_values("date").tail(14)

            avg_daily_sales = (
                float(product_sales["units_sold"].mean())
                if not product_sales.empty else 0.0
            )

            stock = int(product["stock"])
            lead_days = int(product["lead_days"])

            target_stock = int(np.ceil(
                avg_daily_sales * (lead_days + 7)
            ))
            reorder_quantity = max(0, target_stock - stock)

            reasons = []
            if rainy_weather and match["rain"]:
                reasons.append(
                    f"Rain probability reaches {rain_probability:.0f}%"
                )
            if hot_weather and match["heat"]:
                reasons.append(
                    f"Forecast maximum reaches {max_temperature:.1f} °C"
                )

            recommendations.append({
                "Product": name,
                "Weather Match": ", ".join(reasons),
                "Recent Avg Daily Sales": round(avg_daily_sales, 1),
                "Current Stock": stock,
                "Suggested Reorder": reorder_quantity,
                "Planning Note": match["message"]
            })

        st.subheader("Recommended Products")

        if recommendations:
            recommendation_df = pd.DataFrame(recommendations)
            st.dataframe(
                recommendation_df,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"{len(recommendation_df)} product(s) match "
                "the current weather rules."
            )

            st.download_button(
                "Download Weather Recommendation Report",
                recommendation_df.to_csv(index=False).encode("utf-8"),
                file_name="weather_product_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.info(
                "No products matched the current weather rules. "
                "Continue monitoring sales and stock."
            )

        st.caption(
            "Recommendations use illustrative weather thresholds and "
            "product-name/category rules. Reorder quantities are based "
            "on recent sales and the existing inventory formula. "
            "Weather-related demand changes have not been learned or "
            "validated by the ML model."
        )

    except Exception as error:
        st.error(f"Could not generate recommendations: {error}")


# --------------------------------------------------
# SMART BUSINESS ALERTS
# --------------------------------------------------
st.divider()
st.header("🚨 Smart Business Alerts")

alerts = []

# Build inventory recommendation data for Smart Business Alerts
inventory_data = products.copy()

# Calculate recent average daily sales for each product
recent_sales = (
    retail.sort_values("date")
    .groupby("product_id")
    .tail(14)
    .groupby("product_id")["units_sold"]
    .mean()
)

inventory_data["Avg Daily Sales (14d)"] = (
    inventory_data["product_id"]
    .map(recent_sales)
    .fillna(0)
)

# Calculate target stock and suggested reorder quantity
inventory_data["Target Stock"] = np.ceil(
    inventory_data["Avg Daily Sales (14d)"]
    * (inventory_data["lead_days"] + 7)
).astype(int)

inventory_data["Suggested Reorder"] = np.maximum(
    0,
    inventory_data["Target Stock"] - inventory_data["stock"]
)

# Assign inventory status
inventory_data["Status"] = np.where(
    inventory_data["stock"]
    <= inventory_data["Avg Daily Sales (14d)"]
    * inventory_data["lead_days"],
    "URGENT",
    np.where(
        inventory_data["Suggested Reorder"] > 0,
        "REORDER",
        "SUFFICIENT"
    )
)

# Use the column names expected by the alerts section
inventory_data = inventory_data.rename(columns={
    "product_name": "Product",
    "stock": "Current Stock"
})


# Inventory alerts
for _, item in inventory_data.iterrows():
    if item["Status"] == "URGENT":
        alerts.append({
            "Alert": "Critical inventory",
            "Item": item["Product"],
            "Details": (
                f"Stock: {item['Current Stock']} units; "
                f"suggested reorder: {item['Suggested Reorder']} units."
            ),
            "Priority": "High"
        })
    elif item["Status"] == "REORDER":
        alerts.append({
            "Alert": "Reorder recommended",
            "Item": item["Product"],
            "Details": (
                f"Suggested reorder: {item['Suggested Reorder']} units."
            ),
            "Priority": "Medium"
        })

# Customer follow-up alerts using transparent rules
at_risk_customers = customers[
    (customers["days_since_last_purchase"] >= 60)
    | (customers["complaints_last_6m"] >= 2)
]

if not at_risk_customers.empty:
    alerts.append({
        "Alert": "Customer follow-up",
        "Item": f"{len(at_risk_customers)} customers",
        "Details": (
            "Customers have been inactive for 60+ days "
            "or have at least 2 complaints."
        ),
        "Priority": "Medium"
    })

# Low customer rating alerts
low_rating_count = int((feedback["rating"] <= 2).sum())

if low_rating_count > 0:
    alerts.append({
        "Alert": "Poor customer feedback",
        "Item": f"{low_rating_count} reviews",
        "Details": "Review feedback rated 1 or 2 out of 5.",
        "Priority": "Medium"
    })

if alerts:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)
else:
    st.success("No business alerts based on the current rules.")

st.caption(
    "Customer and feedback alerts use simple rules, not a separate "
    "trained prediction model. Review alerts before taking action."
)

st.divider()
st.subheader("📥 Download Unified Retail Data")

st.download_button(
    label="Download Unified Dataset (CSV)",
    data=retail.to_csv(index=False).encode("utf-8"),
    file_name="unified_retail_data.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")
