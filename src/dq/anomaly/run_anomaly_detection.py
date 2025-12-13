from sqlalchemy import text
from src.config.db_config import get_engine
from src.dq.anomaly.detectors import (
    run_isolation_forest_for_table,
    run_dbscan_for_table,
    run_knn_outlier_for_table,
)
from src.dq.imputation.knn_imputation import run_knn_imputation_for_table
from src.dq.validation.run_validation import run_validation_for_table


def reset_anomaly_tables():
    """Clear anomalies and metrics to avoid duplicate rows on rerun."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dq.anomalies RESTART IDENTITY"))
        conn.execute(text("TRUNCATE TABLE dq.anomaly_metrics RESTART IDENTITY"))
    print("Cleared dq.anomalies and dq.anomaly_metrics.")


# 1) UK E-Commerce dataset: raw.ecommerce_transactions
def run_for_ecommerce():
    schema = "raw"
    table = "ecommerce_transactions"
    processed_schema = "processed"

    numeric_cols = ["price", "quantity"]  # adjust if you add total_amount column later

    run_isolation_forest_for_table(
        schema=schema,
        table=table,
        record_id_col="transaction_no",
        entity_id_col="product_no",
        numeric_cols=numeric_cols,
        model_name="isolation_forest_ecommerce"
    )

    run_dbscan_for_table(
        schema=schema,
        table=table,
        id_cols=["transaction_no", "product_no"],
        numeric_cols=numeric_cols,
        limit_rows=50000,
        eps=0.6,
        min_samples=15,
        model_name="dbscan_ecommerce"
    )

    run_knn_outlier_for_table(
        schema=schema,
        table=table,
        id_cols=["transaction_no", "product_no"],
        numeric_cols=numeric_cols,
        k=5,
        contamination=0.02,
        limit_rows=100000,
        model_name="knn_ecommerce"
    )

    run_knn_imputation_for_table(
        schema=schema,  # read from raw, write to processed
        table=table,
        numeric_cols=numeric_cols,
        id_col="transaction_no",
        processed_schema=processed_schema,
        n_neighbors=5,
        limit_rows=100000
    )

    run_validation_for_table(
        schema=processed_schema,
        table=table,
        id_col="transaction_no",
        date_col=None,
        numeric_cols=numeric_cols,
        stock_code_col=None,
        text_cols=["product_name"],
        source_label=f"{processed_schema}.{table}"
    )


# 2) Online Retail dataset: raw.online_retail
def run_for_online_retail():
    schema = "raw"
    table = "online_retail"
    processed_schema = "processed"

    numeric_cols = ["unit_price", "quantity"]  # you can add total_value = unit_price * quantity later

    run_isolation_forest_for_table(
        schema=schema,
        table=table,
        record_id_col="invoice_no",
        entity_id_col="stock_code",
        context_cols=["customer_id"],
        numeric_cols=numeric_cols,
        model_name="isolation_forest_online_retail"
    )

    run_dbscan_for_table(
        schema=schema,
        table=table,
        id_cols=["invoice_no", "stock_code", "customer_id"],
        numeric_cols=numeric_cols,
        limit_rows=50000,
        eps=0.6,
        min_samples=15,
        model_name="dbscan_online_retail"
    )

    run_knn_outlier_for_table(
        schema=schema,
        table=table,
        id_cols=["invoice_no", "stock_code", "customer_id"],
        numeric_cols=numeric_cols,
        k=5,
        contamination=0.02,
        limit_rows=100000,
        model_name="knn_online_retail"
    )

    run_knn_imputation_for_table(
        schema=schema,  # read from raw, write to processed
        table=table,
        numeric_cols=numeric_cols,
        id_col="invoice_no",
        processed_schema=processed_schema,
        n_neighbors=5,
        limit_rows=100000
    )

    run_validation_for_table(
        schema=processed_schema,
        table=table,
        id_col="invoice_no",
        date_col=None,
        numeric_cols=numeric_cols,
        stock_code_col="stock_code",
        text_cols=["description"],
        source_label=f"{processed_schema}.{table}"
    )


if __name__ == "__main__":
    reset_anomaly_tables()
    print("Starting anomaly detection pipeline...")
    run_for_ecommerce()
    run_for_online_retail()
    print("\nAnomaly detection completed for both datasets.")
