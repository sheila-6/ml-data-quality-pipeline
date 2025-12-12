from src.dq.anomaly.detectors import run_isolation_forest_for_table

# 1) UK E-Commerce dataset: raw.ecommerce_transactions
def run_for_ecommerce():
    schema = "raw"
    table = "ecommerce_transactions"

    id_cols = ["transaction_no", "product_no"]   # identifiers for tracking
    numeric_cols = ["price", "quantity"]         # adjust if you add total_amount column later

    run_isolation_forest_for_table(
        schema=schema,
        table=table,
        id_cols=id_cols,
        numeric_cols=numeric_cols,
        model_name="isolation_forest_ecommerce"
    )


# 2) Online Retail dataset: raw.online_retail
def run_for_online_retail():
    schema = "raw"
    table = "online_retail"

    id_cols = ["invoice_no", "stock_code", "customer_id"]
    numeric_cols = ["unit_price", "quantity"]    # you can add total_value = unit_price * quantity later

    run_isolation_forest_for_table(
        schema=schema,
        table=table,
        id_cols=id_cols,
        numeric_cols=numeric_cols,
        model_name="isolation_forest_online_retail"
    )


if __name__ == "__main__":
    print("🚀 Starting anomaly detection pipeline...")
    run_for_ecommerce()
    run_for_online_retail()
    print("\n🎉 Anomaly detection completed for both datasets.")
