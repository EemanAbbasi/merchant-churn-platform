
"""
Stage 2: Merchant Feature Pipeline
Transforms raw Olist CSVs into an analysis-ready merchant-level feature table.
Uses DuckDB to run SQL directly on dataframes.
"""

import duckdb
import pandas as pd
import os

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────────

DATA_DIR = "data/"

print("Loading raw Olist CSVs...")

orders       = pd.read_csv(os.path.join(DATA_DIR, "olist_orders_dataset.csv"), parse_dates=[
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date"
                ])
items        = pd.read_csv(os.path.join(DATA_DIR, "olist_order_items_dataset.csv"))
reviews      = pd.read_csv(os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv"))
payments     = pd.read_csv(os.path.join(DATA_DIR, "olist_order_payments_dataset.csv"))
sellers      = pd.read_csv(os.path.join(DATA_DIR, "olist_sellers_dataset.csv"))
products     = pd.read_csv(os.path.join(DATA_DIR, "olist_products_dataset.csv"))
categories   = pd.read_csv(os.path.join(DATA_DIR, "product_category_name_translation.csv"))

print("All CSVs loaded.")

# ── 2. REGISTER TABLES IN DUCKDB ─────────────────────────────────────────────

con = duckdb.connect()

con.register("orders",    orders)
con.register("items",     items)
con.register("reviews",   reviews)
con.register("payments",  payments)
con.register("sellers",   sellers)
con.register("products",  products)
con.register("categories",categories)

# ── 3. BUILD MERCHANT FEATURE TABLE ──────────────────────────────────────────
# For each seller we compute:
#   - order volume and revenue
#   - average review score
#   - average delivery delay (actual vs estimated)
#   - unique product categories sold
#   - days active on platform
#   - churn label: 1 if seller has no orders in the last 6 months of the dataset

print("Building merchant feature table...")

merchant_features = con.execute("""

    WITH

    -- Observation window boundary
    obs AS (
        SELECT
            MAX(order_purchase_timestamp)                        AS last_date,
            MAX(order_purchase_timestamp) - INTERVAL 6 MONTH    AS churn_cutoff
        FROM orders
        WHERE order_status = 'delivered'
    ),

    -- Join orders to items to get seller-level order data
    seller_orders AS (
        SELECT
            i.seller_id,
            o.order_id,
            o.order_purchase_timestamp                          AS order_date,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,
            i.price,
            i.freight_value,
            i.product_id
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered'
    ),

    -- Seller revenue and volume metrics
    seller_metrics AS (
        SELECT
            seller_id,
            COUNT(DISTINCT order_id)                            AS total_orders,
            SUM(price + freight_value)                         AS total_revenue,
            AVG(price)                                          AS avg_order_value,
            COUNT(DISTINCT product_id)                          AS unique_products,
            MIN(order_date)                                     AS first_order_date,
            MAX(order_date)                                     AS last_order_date,
            DATEDIFF('day', MIN(order_date), MAX(order_date))  AS days_active
        FROM seller_orders
        GROUP BY seller_id
    ),

    -- Average review score per seller
    seller_reviews AS (
        SELECT
            i.seller_id,
            AVG(r.review_score)                                 AS avg_review_score,
            COUNT(r.review_id)                                  AS total_reviews
        FROM reviews r
        JOIN items i ON r.order_id = i.order_id
        GROUP BY i.seller_id
    ),

    -- Average delivery delay in days (positive = late)
    seller_delivery AS (
        SELECT
            seller_id,
            AVG(
                DATEDIFF('day',
                    order_estimated_delivery_date,
                    order_delivered_customer_date)
            )                                                   AS avg_delivery_delay_days
        FROM seller_orders
        WHERE order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
        GROUP BY seller_id
    ),

    -- Most common product category per seller
    seller_category AS (
        SELECT
            so.seller_id,
            COALESCE(c.product_category_name_english, p.product_category_name, 'unknown')
                                                                AS top_category,
            COUNT(*)                                            AS category_count
        FROM seller_orders so
        JOIN products p ON so.product_id = p.product_id
        LEFT JOIN categories c ON p.product_category_name = c.product_category_name
        GROUP BY so.seller_id, top_category
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY so.seller_id
            ORDER BY COUNT(*) DESC
        ) = 1
    ),

    -- Churn label: 1 if seller placed NO orders after churn_cutoff
    churn_labels AS (
        SELECT
            sm.seller_id,
            CASE
                WHEN sm.last_order_date < obs.churn_cutoff THEN 1
                ELSE 0
            END                                                 AS churned,
            DATEDIFF('day', sm.first_order_date, obs.last_date) AS tenure_days
        FROM seller_metrics sm
        CROSS JOIN obs
    )

    -- Final merchant feature table
    SELECT
        sm.seller_id,
        s.seller_state,
        sm.total_orders,
        ROUND(sm.total_revenue, 2)                              AS total_revenue,
        ROUND(sm.avg_order_value, 2)                            AS avg_order_value,
        sm.unique_products,
        sm.days_active,
        cl.tenure_days,
        ROUND(sr.avg_review_score, 2)                           AS avg_review_score,
        sr.total_reviews,
        ROUND(sd.avg_delivery_delay_days, 2)                    AS avg_delivery_delay_days,
        sc.top_category,
        cl.churned

    FROM seller_metrics sm
    JOIN sellers s          ON sm.seller_id    = s.seller_id
    LEFT JOIN seller_reviews sr   ON sm.seller_id = sr.seller_id
    LEFT JOIN seller_delivery sd  ON sm.seller_id = sd.seller_id
    LEFT JOIN seller_category sc  ON sm.seller_id = sc.seller_id
    JOIN churn_labels cl    ON sm.seller_id    = cl.seller_id

    ORDER BY sm.total_orders DESC

""").df()

# ── 4. VALIDATE & SAVE ────────────────────────────────────────────────────────

print("\n── Merchant Feature Table ──────────────────────────────")
print(f"  Total merchants:   {len(merchant_features):,}")
print(f"  Churned:           {merchant_features['churned'].sum():,} ({merchant_features['churned'].mean()*100:.1f}%)")
print(f"  Active:            {(merchant_features['churned']==0).sum():,}")
print(f"  Features:          {merchant_features.shape[1]}")
print(f"\n  Columns: {list(merchant_features.columns)}")
print(f"\n  Sample:\n{merchant_features.head(3).to_string()}")
print("────────────────────────────────────────────────────────\n")

# Check for nulls
null_summary = merchant_features.isnull().sum()
null_cols = null_summary[null_summary > 0]
if len(null_cols) > 0:
    print("  Columns with nulls:")
    print(null_cols)
else:
    print("  No nulls found.")

# Save to CSV for use in modeling + dashboard
os.makedirs("data/processed", exist_ok=True)
merchant_features.to_csv("data/processed/merchant_features.csv", index=False)
print("\nSaved to data/processed/merchant_features.csv")
