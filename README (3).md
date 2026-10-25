# Smart Retail Analytics & Prediction

An interactive retail decision-support prototype combining **Data Mining
and Visualization (DMV)**, **Machine Learning (ML)**, live weather
information, and rule-based business recommendations.

> **Note:** This educational prototype uses synthetic sample data. Its
> results demonstrate the workflow and are not validated real-world
> forecasts.

## Overview

Retailers often keep sales, product, customer, and feedback data in
separate files. This project brings those sources together in a
Streamlit dashboard to help users explore sales trends, forecast product
demand, estimate customer churn risk, review feedback, and make
inventory-planning decisions.

## Problem Statement

Manual analysis of disconnected retail datasets can make it difficult to
identify trends, plan inventory, estimate demand, and recognise
customers who may stop purchasing. The project demonstrates how data
integration, visualization, machine learning, and transparent rules can
support better-informed decisions.

## Objectives

-   Integrate sales, product, customer, and feedback datasets from
    different formats.
-   Clean and prepare data for analysis.
-   Visualize sales trends, product revenue, stock levels, and customer
    ratings.
-   Forecast demand for the next seven days.
-   Estimate customer churn risk.
-   Display live weather and short-term forecasts.
-   Generate inventory recommendations and business alerts.
-   Export unified data and inventory reports as CSV files.

## Features

### Sales Analytics

-   KPI cards for total revenue, units sold, average daily revenue, and
    products represented.
-   Daily revenue line chart.
-   Product-wise revenue bar chart.
-   Product and date-range filters.

### Product Demand Forecasting

A Scikit-learn `RandomForestRegressor` forecasts daily units sold for a
selected product over the next seven days. Current features include: -
Day number in the historical series. - Day of the week. - Rolling
seven-day average of demand.

The dashboard displays forecast values and a historical test Mean
Absolute Error (MAE).

### Customer Churn Analysis

A Scikit-learn `RandomForestClassifier` estimates the churned-class
score using: - Customer tenure in months. - Monthly spending. -
Purchases in the last 90 days. - Days since the last purchase. -
Complaints in the last six months.

The dashboard reports test accuracy and displays customers with the
highest predicted churn scores. Scores are estimates, not certainties.

### Customer Feedback

-   Average customer rating.
-   Rating distribution chart.
-   Recent feedback table.
-   Alerts for ratings of 1 or 2 out of 5.

### Weather-Aware Retail Insights

The application calls the Open-Meteo API for current weather and a
three-day forecast for a selected location. Predefined rules generate
suggestions to review seasonal products when rain probability or
temperature crosses a threshold.

**Weather is not currently an input to the trained ML demand model.**
Weather suggestions are rule-based, not weather-based ML predictions.

### Smart Inventory Recommendations

The application calculates average daily sales over the latest 14 days
for each product and estimates target stock:

`Target Stock = ceil(Average Daily Demand × (Supplier Lead Time + 7))`

`Suggested Reorder = max(0, Target Stock − Current Stock)`

A product is marked urgent when current stock is less than or equal to
average daily demand multiplied by supplier lead time. Otherwise, a
positive reorder quantity is marked for reorder; the remaining products
are marked sufficient.

These are planning estimates, not guaranteed demand or automatic
purchase orders.

### Smart Business Alerts

Rule-based alerts flag: - Urgent inventory and products recommended for
reorder. - Customers inactive for at least 60 days or with at least two
complaints. - Feedback rated 1 or 2 out of 5.

These alerts are not separate trained ML models.

### Downloadable Reports

-   Unified retail dataset as CSV.
-   Inventory reorder report as CSV.

## How the System Works

1.  **Data ingestion:** Read sales CSV, product Excel, customer CSV, and
    feedback JSON.
2.  **Preprocessing:** Remove duplicates, parse dates, inspect missing
    values, and prepare consistent columns.
3.  **Integration:** Join sales and product records using `product_id`.
4.  **Analysis and visualization:** Aggregate sales and revenue, then
    display interactive charts and tables.
5.  **Machine learning:** Train a demand regressor and churn classifier.
6.  **Decision support:** Calculate inventory suggestions and create
    alerts using explicit rules.
7.  **Weather integration:** Request current conditions and forecasts
    from Open-Meteo.
8.  **Reporting:** Display results and allow CSV downloads.

## DMV and Machine Learning

### Data Mining and Visualization (DMV)

The project demonstrates: - Time-based sales and revenue aggregation. -
Product performance comparisons. - Inventory summaries. - Customer
behaviour analysis. - Feedback rating distributions. - Interactive
visualizations and KPI cards.

Pandas handles data manipulation and aggregation. Plotly creates
interactive charts, while Streamlit presents the dashboard.

### Machine Learning

The project uses supervised learning for two tasks:

  -----------------------------------------------------------------------
  Task              Algorithm         Type              Target
  ----------------- ----------------- ----------------- -----------------
  Product demand    Random Forest     Regression        Daily units sold
  forecasting       Regressor                           

  Customer churn    Random Forest     Classification    `churned` label
  prediction        Classifier                          
  -----------------------------------------------------------------------

Regression predicts a numeric demand value. Classification predicts a
churn class and provides a score for the churned class.

## Technology Stack

-   **Python:** Main programming language.
-   **Pandas:** Data loading, cleaning, joining, and aggregation.
-   **NumPy:** Numerical operations and sample-data generation.
-   **Scikit-learn:** Random Forest models and evaluation metrics.
-   **Streamlit:** Interactive web application.
-   **Plotly Express:** Interactive charts.
-   **OpenPyXL:** Excel file support.
-   **Requests:** HTTP requests to the weather API.
-   **CSV, Excel, JSON:** Data formats.
-   **Open-Meteo:** Weather forecast API.

## Project Structure

``` text
SmartRetailFromScratch/
├── app.py
├── create_data.py
├── data_analysis.py
├── data/
│   ├── sales.csv
│   ├── products.xlsx
│   ├── customers.csv
│   ├── feedback.json
│   └── unified_retail_data.csv
└── README.md
```

The `.venv/` virtual environment may also exist locally and is normally
excluded from version control.

## Installation and Setup

### Requirements

-   Python 3.10 or newer recommended.
-   Internet access for live weather data.
-   Terminal and an editor such as Visual Studio Code.

### 1. Open the project folder

``` bash
cd ~/Documents/SmartRetailFromScratch
```

### 2. Create and activate a virtual environment (macOS/Linux)

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn plotly streamlit openpyxl requests
```

### 4. Generate the sample datasets

``` bash
python create_data.py
```

This creates `sales.csv`, `products.xlsx`, `customers.csv`, and
`feedback.json` in the `data/` directory.

### 5. Run the data analysis script

``` bash
python data_analysis.py
```

This loads and inspects the datasets, performs basic cleaning, joins
sales with product information, calculates revenue, and writes
`data/unified_retail_data.csv`.

## Run the Application

From the project root, with the virtual environment activated:

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`. To stop the application, press `Ctrl+C` in the
terminal.

## Dataset Description

The project uses generated sample data:

  ------------------------------------------------------------------------
  Dataset                           Approximate size Example fields
  --------------------- ---------------------------- ---------------------
  `sales.csv`                               600 rows Date, product ID,
                                                     units sold, promotion

  `products.xlsx`                         5 products Product name,
                                                     category, unit price,
                                                     stock, supplier lead
                                                     time

  `customers.csv`                      250 customers Tenure, spend,
                                                     purchase frequency,
                                                     recency, complaints,
                                                     churn label

  `feedback.json`                        100 records Customer ID, product
                                                     ID, rating, comment,
                                                     date
  ------------------------------------------------------------------------

These values and labels are synthetic, not records from a real retailer.

## Model Evaluation and Limitations

-   MAE and accuracy are computed on the current synthetic sample; they
    do not establish real-world performance.
-   A stronger evaluation should include baseline comparisons,
    precision, recall, F1-score, a confusion matrix, and suitable
    validation.
-   Time-series features should be constructed so each test prediction
    uses only information available at that time. Rolling-average
    features should be checked for leakage during rigorous evaluation.
-   Churn scores are model outputs, not guarantees of future behaviour.
-   Inventory calculations use a fixed seven-day buffer and do not model
    demand uncertainty, minimum order quantities, or all supplier
    constraints.
-   Weather suggestions use predefined thresholds and do not feed
    weather into the ML model.
-   Revenue is not profit; product costs and operating expenses are not
    included.
-   The prototype reads local files and is not a multi-user production
    system.

## Future Enhancements

-   Train and validate models on real retail data.
-   Add historical weather features to demand forecasting and test
    whether they improve performance.
-   Add market-basket analysis using transaction-level basket IDs.
-   Compare Random Forest against baseline and alternative forecasting
    models.
-   Add precision, recall, F1-score, confusion matrix, and
    feature-importance views.
-   Improve inventory planning with safety stock, demand uncertainty,
    and supplier constraints.
-   Add persistent database storage, authentication, and role-based
    access.
-   Add validated sentiment analysis for customer feedback.
-   Deploy with suitable security and reliability controls.

## Summary

**Smart Retail Analytics & Prediction** is an educational prototype
combining DMV, supervised machine learning, live weather integration,
rule-based alerts, and inventory decision support in one interactive
dashboard. It demonstrates how different retail data sources can be
transformed into insights and estimates that support business decisions,
while real-world validation remains an important next step.
