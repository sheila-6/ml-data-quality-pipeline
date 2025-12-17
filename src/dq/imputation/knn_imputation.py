import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from src.config.db_config import get_engine


def run_knn_imputation_for_table(
    schema: str,
    table: str,
    numeric_cols: list,
    id_col: str | None = None,
    processed_schema: str = "processed",
    n_neighbors: int = 5,
    limit_rows: int | None = None
):
    """
    Perform KNN imputation on numeric columns while preserving anomaly flags.
    """

    engine = get_engine()
    full_table = f"{schema}.{table}"
    processed_table = f"{processed_schema}.{table}"

    print(f"\nRunning KNN imputation on {full_table}")

    # Load raw data (optionally sample to limit runtime)
    if limit_rows:
        query = text(f"SELECT * FROM {full_table} ORDER BY RANDOM() LIMIT :limit_rows")
        df = pd.read_sql(query, engine, params={"limit_rows": limit_rows})
        print(f"Sampled {len(df)} rows (limit {limit_rows}) from {full_table} for imputation.")
    else:
        df = pd.read_sql(f"SELECT * FROM {full_table}", engine)
    if df.empty:
        print("Table empty. Skipping.")
        return

    # Join anomaly flags using provided id_col if available, else fallback to index
    if id_col and id_col in df.columns:
        anomalies = pd.read_sql(
            f"""
            SELECT DISTINCT record_id
            FROM dq.anomalies
            WHERE source_table = '{full_table}'
            """,
            engine
        )
        if not anomalies.empty:
            df["is_anomaly"] = df[id_col].astype(str).isin(anomalies["record_id"].astype(str))
        else:
            df["is_anomaly"] = False
    else:
        df["is_anomaly"] = False

    # Apply business rules → convert invalid values to NaN
    for col in numeric_cols:
        if col in df.columns:
            if "price" in col:
                df.loc[df[col] <= 0, col] = np.nan
            if "quantity" in col:
                df.loc[df[col] < 0, col] = np.nan

    # Track missing counts before
    missing_before = df[numeric_cols].isna().sum()

    # Split normal vs anomaly rows
    normal_df = df[~df["is_anomaly"]].copy()
    anomaly_df = df[df["is_anomaly"]].copy()

    # Scale numeric data
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_df[numeric_cols]) if len(normal_df) else np.empty((0, len(numeric_cols)))
    anomaly_scaled = scaler.transform(anomaly_df[numeric_cols]) if len(anomaly_df) else np.empty((0, len(numeric_cols)))

    # KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    normal_imputed = imputer.fit_transform(normal_scaled) if len(normal_df) else normal_scaled
    anomaly_imputed = imputer.transform(anomaly_scaled) if len(anomaly_df) else anomaly_scaled

    # Reverse scaling
    if len(normal_df):
        normal_df[numeric_cols] = scaler.inverse_transform(normal_imputed)
    if len(anomaly_df):
        anomaly_df[numeric_cols] = scaler.inverse_transform(anomaly_imputed)

    # Recombine
    df_imputed = pd.concat([normal_df, anomaly_df]).sort_index()

    # Track missing after
    missing_after = df_imputed[numeric_cols].isna().sum()

    # Save processed data without dropping the table (avoid breaking dependent views)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {processed_schema}"))
        try:
            conn.execute(text(f"DELETE FROM {processed_schema}.{table}"))
        except Exception:
            # Table may not exist yet; that's fine—pandas will create it.
            pass

    df_imputed.to_sql(
        table,
        con=engine,
        schema=processed_schema,
        if_exists="append",
        index=False
    )

    print(f"Saved imputed data to {processed_table}")

    # Audit log
    audit_rows = []
    for col in numeric_cols:
        audit_rows.append({
            "table_name": processed_table,
            "column_name": col,
            "rows_imputed": int(missing_before[col] - missing_after[col]),
            "method": "knn_imputation"
        })

    audit_df = pd.DataFrame(audit_rows)
    # Ensure audit schema exists
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS audit"))

    audit_df.to_sql(
        "imputation_log",
        con=engine,
        schema="audit",
        if_exists="append",
        index=False
    )

    print("Imputation audit logged.")
