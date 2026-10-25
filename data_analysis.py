
import json
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. DEFINE DATA FOLDER
# --------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

print("\n========== SMART RETAIL DATA ANALYSIS ==========\n")

# --------------------------------------------------
# 2. LOAD ALL FOUR DATASETS
# --------------------------------------------------
print("Step 1: Loading datasets...")

sales = pd.read_csv(DATA_DIR / "sales.csv")
products = pd.read_excel(DATA_DIR / "products.xlsx")
customers = pd.read_csv(DATA_DIR / "customers.csv")

with open(DATA_DIR / "feedback.json", "r", encoding="utf-8") as file:
    feedback = pd.DataFrame(json.load(file))

print("All four datasets loaded successfully!")

# --------------------------------------------------
# 3. DISPLAY INITIAL DATASET INFORMATION
# --------------------------------------------------
print("\nStep 2: Initial dataset sizes")

print(f"Sales records: {len(sales)}")
print(f"Product records: {len(products)}")
print(f"Customer records: {len(customers)}")
print(f"Feedback records: {len(feedback)}")

print("\nSales dataset preview:")
print(sales.head())

print("\nProducts dataset preview:")
print(products.head())

print("\nCustomers dataset preview:")
print(customers.head())

print("\nFeedback dataset preview:")
print(feedback.head())

# --------------------------------------------------
# 4. CHECK MISSING VALUES
# --------------------------------------------------
print("\nStep 3: Missing values before cleaning")

for name, dataset in [
    ("Sales", sales),
    ("Products", products),
    ("Customers", customers),
    ("Feedback", feedback)
]:
    print(f"\n{name}:")
    print(dataset.isnull().sum())

# --------------------------------------------------
# 5. REMOVE DUPLICATE RECORDS
# --------------------------------------------------
print("\nStep 4: Removing duplicate records")

sales = sales.drop_duplicates()
products = products.drop_duplicates(subset=["product_id"])
customers = customers.drop_duplicates(subset=["customer_id"])
feedback = feedback.drop_duplicates(subset=["feedback_id"])

# --------------------------------------------------
# 6. CLEAN AND VALIDATE DATES
# --------------------------------------------------
print("\nStep 5: Cleaning date columns")

sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
feedback["date"] = pd.to_datetime(feedback["date"], errors="coerce")

sales = sales.dropna(subset=["date", "product_id", "units_sold"])
feedback = feedback.dropna(subset=["date"])

sales["units_sold"] = pd.to_numeric(
    sales["units_sold"], errors="coerce"
)
sales = sales.dropna(subset=["units_sold"])
sales["units_sold"] = sales["units_sold"].astype(int)

# --------------------------------------------------
# 7. INTEGRATE SALES AND PRODUCT DATA
# --------------------------------------------------
print("\nStep 6: Combining sales and product data")

retail_data = sales.merge(
    products,
    on="product_id",
    how="left",
    validate="many_to_one",
    indicator=True
)

unmatched = (retail_data["_merge"] != "both").sum()

if unmatched > 0:
    print(f"Warning: {unmatched} sales records have no matching product.")
else:
    print("Every sales record matches a product.")

retail_data = retail_data.drop(columns=["_merge"])

# Remove records without essential product details
retail_data = retail_data.dropna(
    subset=["product_name", "unit_price"]
)

# --------------------------------------------------
# 8. CALCULATE REVENUE
# --------------------------------------------------
print("\nStep 7: Calculating revenue")

retail_data["revenue"] = (
    retail_data["units_sold"] * retail_data["unit_price"]
)

# --------------------------------------------------
# 9. GENERATE BUSINESS SUMMARY
# --------------------------------------------------
print("\n========== BUSINESS SUMMARY ==========")

total_units = retail_data["units_sold"].sum()
total_revenue = retail_data["revenue"].sum()
average_daily_revenue = (
    retail_data.groupby("date")["revenue"].sum().mean()
)

print(f"Total units sold: {total_units:,}")
print(f"Total revenue: Rs. {total_revenue:,.2f}")
print(f"Average daily revenue: Rs. {average_daily_revenue:,.2f}")

print("\nRevenue by product:")

product_summary = (
    retail_data.groupby("product_name")
    .agg(
        units_sold=("units_sold", "sum"),
        total_revenue=("revenue", "sum")
    )
    .sort_values("total_revenue", ascending=False)
)

print(product_summary)

# --------------------------------------------------
# 10. SAVE THE UNIFIED DATASET
# --------------------------------------------------
output_path = DATA_DIR / "unified_retail_data.csv"

retail_data.to_csv(output_path, index=False)

print("\n========== ANALYSIS COMPLETED ==========")
print(f"Unified dataset saved to: {output_path}")
print(f"Final unified records: {len(retail_data)}")
print("\nYou can now use this dataset to build dashboards")
print("and demand prediction models.")
