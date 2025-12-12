import json
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.config.db_config import get_engine


def run_isolation_forest_for_table(
    schema: str,
    table: str,
    record_id_col: str | None = None,
    entity_id_col: str | None = None,
    numeric_cols: list | None = None,
    context_cols: list | None = None,
    model_name: str = "isolation_forest"
):
    """
    Run IsolationForest anomaly detection for a given table and save results
    into dq.anomalies and dq.anomaly_metrics.
    """

    engine = get_engine()
    full_table = f"{schema}.{table}"

    print(f"\nRunning IsolationForest on {full_table} ...")

    # Load data
    df = pd.read_sql(f"SELECT * FROM {full_table}", engine)
    if df.empty:
        print(f"{full_table} is empty. Skipping.")
        return

    # Select numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64", "Float64"]).columns.tolist()

    if not numeric_cols:
        print(f"No numeric columns found in {full_table}. Cannot run anomaly detection.")
        return

    print(f"Using numeric columns for anomaly detection: {numeric_cols}")
    X = df[numeric_cols].copy()

    # Handle missing values - fill NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    # Fit IsolationForest
    model = IsolationForest(
        contamination=0.02,
        random_state=42,
        n_estimators=200
    )
    model.fit(X)

    # Get anomaly scores & labels
    scores = model.decision_function(X)
    labels = model.predict(X)  # -1 = anomaly, 1 = normal

    df["anomaly_score"] = scores
    df["is_anomaly"] = (labels == -1)

    total_rows = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_rate = anomaly_count / total_rows if total_rows > 0 else 0

    print(f"Total rows: {total_rows}, anomalies detected: {anomaly_count} ({anomaly_rate:.2%})")

    # Prepare anomalies DataFrame to save
    anomalies = df[df["is_anomaly"]].copy()

    context_cols = context_cols or []

    # Ensure required ID/context columns exist even if missing in source
    for col in [record_id_col, entity_id_col] + context_cols:
        if col and col not in anomalies.columns:
            print(f"Warning: missing column '{col}' in {full_table}; filling with None.")
            anomalies[col] = None

    anomalies["record_id"] = anomalies[record_id_col] if record_id_col else None
    anomalies["entity_id"] = anomalies[entity_id_col] if entity_id_col else None

    # Build anomaly_context JSON from optional extra columns
    if context_cols:
        context_payloads = []
        for row in anomalies[context_cols].to_dict(orient="records"):
            cleaned = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            context_payloads.append(None if all(v is None for v in cleaned.values()) else json.dumps(cleaned))
        anomalies["anomaly_context"] = context_payloads
    else:
        anomalies["anomaly_context"] = None

    anomalies["source_table"] = full_table
    anomalies["anomaly_type"] = model_name

    # Reorder columns to match dq.anomalies schema
    anomalies = anomalies[
        ["source_table", "record_id", "entity_id", "anomaly_type", "anomaly_score", "anomaly_context"]
    ]

    # Write anomalies into dq.anomalies
    anomalies.to_sql(
        "anomalies",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )
    print(f"Saved {len(anomalies)} anomalies into dq.anomalies")

    # Save metrics
    metrics_df = pd.DataFrame(
        [{
            "source_table": full_table,
            "model_name": model_name,
            "total_rows": total_rows,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate
        }]
    )

    metrics_df.to_sql(
        "anomaly_metrics",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )
    print(f"Saved summary metrics into dq.anomaly_metrics")
