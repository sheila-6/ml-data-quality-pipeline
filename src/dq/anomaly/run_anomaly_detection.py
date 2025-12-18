import logging
from pathlib import Path

from sqlalchemy import text
from src.config.db_config import get_engine
from src.dq.anomaly.detectors import (
    run_isolation_forest_for_table,
    run_dbscan_for_table,
    run_knn_outlier_for_table,
)
from src.dq.imputation.knn_imputation import run_knn_imputation_for_table
from src.dq.validation.run_validation import run_validation_for_table

LOG_NAME = "dq_pipeline"


def setup_logging(log_file: str = "logs/pipeline.log"):
    """Configure shared pipeline logging to file + console with timestamps."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )
    return logging.getLogger(LOG_NAME)


def deduplicate_processed_table(schema: str, table: str, key_cols: list[str]):
    """
    Drop duplicate rows in-place on processed tables, keeping the first per key.
    """
    logger = logging.getLogger(LOG_NAME)
    engine = get_engine()
    keys = ", ".join(key_cols)
    with engine.begin() as conn:
        duplicates_to_remove = conn.execute(
            text(
                f"""
                SELECT COALESCE(SUM(c - 1), 0) AS dup_rows
                FROM (
                    SELECT COUNT(*) AS c
                    FROM {schema}.{table}
                    GROUP BY {keys}
                ) sub
                """
            )
        ).scalar_one()

        if duplicates_to_remove == 0:
            logger.info("No duplicates found for %s.%s on keys (%s).", schema, table, keys)
            return

        conn.execute(
            text(
                f"""
                DELETE FROM {schema}.{table} a
                USING (
                    SELECT ctid
                    FROM (
                        SELECT ctid, ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY ctid) AS rn
                        FROM {schema}.{table}
                    ) t
                    WHERE rn > 1
                ) d
                WHERE a.ctid = d.ctid
                """
            )
        )

    logger.info(
        "Removed %s duplicate rows from %s.%s using keys (%s).",
        duplicates_to_remove,
        schema,
        table,
        keys,
    )


def reset_anomaly_tables():
    """Clear anomalies and metrics to avoid duplicate rows on rerun."""
    logger = logging.getLogger(LOG_NAME)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE dq.anomalies RESTART IDENTITY"))
        conn.execute(text("TRUNCATE TABLE dq.anomaly_metrics RESTART IDENTITY"))
    logger.info("Cleared dq.anomalies and dq.anomaly_metrics.")


# 1) UK E-Commerce dataset: raw.ecommerce_transactions
def run_for_ecommerce():
    logger = logging.getLogger(LOG_NAME)
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
        limit_rows=50000,  # sample to avoid OOM
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
        limit_rows=100000,  # capped sample
        model_name="knn_ecommerce"
    )

    run_knn_imputation_for_table(
        schema=schema,  # read from raw, write to processed
        table=table,
        numeric_cols=numeric_cols,
        id_col="transaction_no",
        processed_schema=processed_schema,
        n_neighbors=5,
        limit_rows=None  # full data
    )

    deduplicate_processed_table(processed_schema, table, ["transaction_no", "product_no"])

    run_validation_for_table(
        schema=processed_schema,
        table=table,
        id_col="transaction_no",
        date_col=None,
        numeric_cols=numeric_cols,
        stock_code_col=None,
        text_cols=["product_name"],
        duplicate_keys=["transaction_no", "product_no"],
        source_label=f"{processed_schema}.{table}"
    )


# 2) Online Retail dataset: raw.online_retail
def run_for_online_retail():
    logger = logging.getLogger(LOG_NAME)
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
        limit_rows=50000,  # sample to avoid OOM
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
        limit_rows=100000,  # capped sample
        model_name="knn_online_retail"
    )

    run_knn_imputation_for_table(
        schema=schema,  # read from raw, write to processed
        table=table,
        numeric_cols=numeric_cols,
        id_col="invoice_no",
        processed_schema=processed_schema,
        n_neighbors=5,
        limit_rows=None  # full data
    )

    deduplicate_processed_table(processed_schema, table, ["invoice_no", "stock_code"])

    run_validation_for_table(
        schema=processed_schema,
        table=table,
        id_col="invoice_no",
        date_col=None,
        numeric_cols=numeric_cols,
        stock_code_col="stock_code",
        text_cols=["description"],
        duplicate_keys=["invoice_no", "stock_code"],
        source_label=f"{processed_schema}.{table}"
    )


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting anomaly detection pipeline...")
    reset_anomaly_tables()
    try:
        logger.info("Running UK E-Commerce pipeline...")
        run_for_ecommerce()
        logger.info("UK E-Commerce pipeline completed.")
    except Exception:
        logger.exception("UK E-Commerce pipeline failed.")
        raise

    try:
        logger.info("Running Online Retail pipeline...")
        run_for_online_retail()
        logger.info("Online Retail pipeline completed.")
    except Exception:
        logger.exception("Online Retail pipeline failed.")
        raise

    logger.info("Anomaly detection completed for both datasets.")
