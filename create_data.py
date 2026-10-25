
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Create the data folder if it does not exist
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random generator for reproducible data
rng = np.random.default_rng(42)

# --------------------------------------------------
# 1. PRODUCT DATASET (Excel)
# --------------------------------------------------
products = pd.DataFrame([
    {
        "product_id": "P01",
        "product_name": "Bottled Water",
        "category": "Beverages",
        "unit_price": 20,
        "stock": 100,
        "lead_days": 3
    },
    {
        "product_id": "P02",
        "product_name": "Iced Tea",
        "category": "Beverages",
        "unit_price": 45,
        "stock": 40,
        "lead_days": 4
    },
    {
        "product_id": "P03",
        "product_name": "Umbrella",
        "category": "Seasonal",
        "unit_price": 350,
        "stock": 15,
        "lead_days": 7
    },
    {
        "product_id": "P04",
        "product_name": "Sunscreen",
        "category": "Personal Care",
        "unit_price": 220,
        "stock": 25,
        "lead_days": 5
    },
    {
        "product_id": "P05",
        "product_name": "Instant Noodles",
        "category": "Food",
        "unit_price": 30,
        "stock": 60,
        "lead_days": 2
    }
])

products.to_excel(DATA_DIR / "products.xlsx", index=False)

# --------------------------------------------------
# 2. SALES DATASET (CSV)
# 120 days of sales for each product
# --------------------------------------------------
sales_records = []

start_date = date.today() - timedelta(days=119)

base_demand = {
    "P01": 35,
    "P02": 22,
    "P03": 7,
    "P04": 12,
    "P05": 30
}

for day in range(120):
    current_date = start_date + timedelta(days=day)

    for product in products.to_dict("records"):
        product_id = product["product_id"]

        weekend_factor = (
            1.2 if current_date.weekday() >= 5 else 1.0
        )

        promotion = int(rng.random() < 0.15)

        demand = base_demand[product_id]

        units_sold = int(
            rng.poisson(
                demand
                * weekend_factor
                * (1.2 if promotion else 1.0)
            )
        )

        sales_records.append({
            "date": current_date.isoformat(),
            "product_id": product_id,
            "units_sold": units_sold,
            "promotion": promotion
        })

sales = pd.DataFrame(sales_records)
sales.to_csv(DATA_DIR / "sales.csv", index=False)

# --------------------------------------------------
# 3. CUSTOMER DATASET (CSV)
# --------------------------------------------------
customer_records = []

for i in range(1, 251):
    recency = int(rng.integers(1, 100))
    purchases = int(rng.integers(1, 16))
    complaints = int(rng.poisson(0.7))
    tenure = int(rng.integers(1, 49))
    spending = round(float(rng.uniform(300, 4000)), 2)

    # Generate a synthetic churn probability
    risk_score = (
        -2
        + 0.035 * recency
        + 0.55 * complaints
        - 0.08 * purchases
        - 0.018 * tenure
    )

    probability = 1 / (1 + np.exp(-risk_score))
    churned = int(rng.random() < probability)

    customer_records.append({
        "customer_id": f"C{i:03d}",
        "tenure_months": tenure,
        "monthly_spend": spending,
        "purchases_last_90d": purchases,
        "days_since_last_purchase": recency,
        "complaints_last_6m": complaints,
        "churned": churned
    })

customers = pd.DataFrame(customer_records)
customers.to_csv(DATA_DIR / "customers.csv", index=False)

# --------------------------------------------------
# 4. CUSTOMER FEEDBACK DATASET (JSON)
# --------------------------------------------------
comments = [
    (5, "Excellent product and quick delivery"),
    (4, "Good value for money"),
    (3, "Average product quality"),
    (2, "Delivery was late"),
    (1, "Damaged packaging and poor experience")
]

feedback_records = []

for i in range(1, 101):
    rating, comment = comments[int(rng.integers(0, len(comments)))]

    product_id = str(rng.choice(products["product_id"]))

    customer_number = int(rng.integers(1, 251))

    feedback_records.append({
        "feedback_id": f"F{i:03d}",
        "customer_id": f"C{customer_number:03d}",
        "product_id": product_id,
        "rating": rating,
        "comment": comment,
        "date": (
            date.today()
            - timedelta(days=int(rng.integers(0, 90)))
        ).isoformat()
    })

with open(
    DATA_DIR / "feedback.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feedback_records, file, indent=2)

# --------------------------------------------------
# 5. CONFIRM DATASET CREATION
# --------------------------------------------------
print("\nAll four datasets created successfully!")

print(f"Sales records: {len(sales)}")
print(f"Products: {len(products)}")
print(f"Customers: {len(customers)}")
print(f"Feedback records: {len(feedback_records)}")

print("\nFiles saved in:", DATA_DIR.resolve())

for filename in [
    "sales.csv",
    "products.xlsx",
    "customers.csv",
    "feedback.json"
]:
    file_path = DATA_DIR / filename
    print(f"Created: {file_path.name} ({file_path.stat().st_size} bytes)")
