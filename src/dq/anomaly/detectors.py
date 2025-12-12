import json
import pandas as pd
from sqlalchemy import text
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from src.config.db_config import get_engine


def _prepare_anomaly_context(df: pd.DataFrame, context_cols: list | None) -> list:
    """Serialize optional context columns to JSON strings or None."""
    if not context_cols:
        return [None for _ in range(len(df))]

    payloads = []
    for row in df[context_cols].to_dict(orient="records"):
        cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        payloads.append(None if all(v is None for v in cleaned.values()) else json.dumps(cleaned))
    return payloads


def run_isolation_forest_for_table(
    schema: str,
    table: str,
    record_id_col: str,
    entity_id_col: str,
    numeric_cols: list | None = None,
    context_cols: list | None = None,
    model_name: str = "isolation_forest"
):
    """
    Run IsolationForest anomaly detection for a given table and save results
    into dq.anomalies and dq.anomaly_metrics using the generic schema.
    """

    engine = get_engine()
    full_table = f"{schema}.{table}"

    print(f"\nRunning IsolationForest on {full_table} ...")

    df = pd.read_sql(f"SELECT * FROM {full_table}", engine)
    if df.empty:
        print(f"{full_table} is empty. Skipping.")
        return

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()

    if not numeric_cols:
        print(f"No numeric columns found in {full_table}. Cannot run anomaly detection.")
        return

    print(f"Using numeric columns for anomaly detection: {numeric_cols}")
    X = df[numeric_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    model = IsolationForest(contamination=0.02, random_state=42, n_estimators=200)
    model.fit(X)

    scores = model.decision_function(X)
    labels = model.predict(X)  # -1 = anomaly, 1 = normal

    df["anomaly_score"] = scores
    df["is_anomaly"] = labels == -1

    total_rows = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_rate = anomaly_count / total_rows if total_rows else 0

    print(f"Total rows: {total_rows}, anomalies detected: {anomaly_count} ({anomaly_rate:.2%})")

    anomalies_df = df[df["is_anomaly"]].copy()

    # Ensure ID/context columns exist
    for col in [record_id_col, entity_id_col] + (context_cols or []):
        if col and col not in anomalies_df.columns:
            print(f"Warning: missing column '{col}' in {full_table}; filling with None.")
            anomalies_df[col] = None

    context_payloads = _prepare_anomaly_context(anomalies_df, context_cols)

    anomalies = pd.DataFrame({
        "source_table": full_table,
        "record_id": anomalies_df[record_id_col].astype(str),
        "entity_id": anomalies_df[entity_id_col].astype(str),
        "anomaly_type": model_name,
        "anomaly_score": anomalies_df["anomaly_score"],
        "anomaly_context": context_payloads
    })

    anomalies.to_sql(
        "anomalies",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )

    print(f"Saved {len(anomalies)} anomalies into dq.anomalies")

    metrics_df = pd.DataFrame([{
        "source_table": full_table,
        "model_name": model_name,
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate
    }])

    metrics_df.to_sql(
        "anomaly_metrics",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )

    print("Saved summary metrics into dq.anomaly_metrics")


def run_dbscan_for_table(
    schema: str,
    table: str,
    id_cols: list,
    numeric_cols: list | None = None,
    limit_rows: int | None = None,
    eps: float = 0.5,
    min_samples: int = 10,
    model_name: str = "dbscan"
):
    """
    Run DBSCAN-based anomaly detection for density-based outliers.
    """

    engine = get_engine()
    full_table = f"{schema}.{table}"

    print(f"\nRunning DBSCAN on {full_table} ...")

    # Load data (optionally sample to limit runtime)
    if limit_rows:
        query = text(f"SELECT * FROM {full_table} ORDER BY RANDOM() LIMIT :limit_rows")
        df = pd.read_sql(query, engine, params={"limit_rows": limit_rows})
        print(f"Sampled {len(df)} rows (limit {limit_rows}) from {full_table} for DBSCAN.")
    else:
        df = pd.read_sql(f"SELECT * FROM {full_table}", engine)

    if df.empty:
        print(f"{full_table} is empty. Skipping.")
        return

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()

    if not numeric_cols:
        print(f"No numeric columns found in {full_table}. Cannot run anomaly detection.")
        return

    print(f"Using numeric columns for anomaly detection: {numeric_cols}")
    X = df[numeric_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)

    df["is_anomaly"] = labels == -1  # DBSCAN marks noise as -1
    df["anomaly_score"] = -1.0       # placeholder score for DBSCAN noise points

    total_rows = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_rate = anomaly_count / total_rows if total_rows else 0

    print(f"Total rows: {total_rows}, anomalies detected: {anomaly_count} ({anomaly_rate:.2%})")

    anomalies_df = df[df["is_anomaly"]].copy()

    # Map id columns: first -> record_id, second -> entity_id, rest -> context
    record_id_col = id_cols[0] if len(id_cols) > 0 else None
    entity_id_col = id_cols[1] if len(id_cols) > 1 else None
    context_cols = id_cols[2:] if len(id_cols) > 2 else []

    for col in [record_id_col, entity_id_col] + context_cols:
        if col and col not in anomalies_df.columns:
            print(f"Warning: missing column '{col}' in {full_table}; filling with None.")
            anomalies_df[col] = None

    record_ids = anomalies_df[record_id_col].astype(str) if record_id_col else None
    entity_ids = anomalies_df[entity_id_col].astype(str) if entity_id_col else None
    context_payloads = _prepare_anomaly_context(anomalies_df, context_cols)

    anomalies = pd.DataFrame({
        "source_table": full_table,
        "record_id": record_ids,
        "entity_id": entity_ids,
        "anomaly_type": model_name,
        "anomaly_score": anomalies_df["anomaly_score"],
        "anomaly_context": context_payloads
    })

    anomalies.to_sql(
        "anomalies",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )

    print(f"Saved {len(anomalies)} DBSCAN anomalies into dq.anomalies")

    metrics_df = pd.DataFrame([{
        "source_table": full_table,
        "model_name": model_name,
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate
    }])

    metrics_df.to_sql(
        "anomaly_metrics",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )

    print("Saved summary metrics into dq.anomaly_metrics")
