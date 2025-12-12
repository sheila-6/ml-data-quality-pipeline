import pandas as pd
from sklearn.ensemble import IsolationForest
from src.config.db_config import get_engine


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
    into dq.anomalies and dq.anomaly_metrics using a generic schema.
    """

    engine = get_engine()
    full_table = f"{schema}.{table}"

    print(f"\n🔎 Running IsolationForest on {full_table} ...")

    # Load data
    df = pd.read_sql(f"SELECT * FROM {full_table}", engine)
    if df.empty:
        print(f"⚠ {full_table} is empty. Skipping.")
        return

    # Select numeric columns if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(
            include=["int64", "float64", "Int64", "Float64"]
        ).columns.tolist()

    if not numeric_cols:
        print(f"⚠ No numeric columns found in {full_table}. Cannot run anomaly detection.")
        return

    print(f"Using numeric columns for anomaly detection: {numeric_cols}")
    X = df[numeric_cols].copy()

    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))

    # Fit Isolation Forest
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
    df["is_anomaly"] = labels == -1

    total_rows = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_rate = anomaly_count / total_rows if total_rows else 0

    print(
        f"Total rows: {total_rows}, "
        f"anomalies detected: {anomaly_count} ({anomaly_rate:.2%})"
    )

    # -----------------------------
    # Prepare anomalies dataframe
    # -----------------------------
    anomalies_df = df[df["is_anomaly"]].copy()

    # Handle missing columns safely
    if record_id_col not in anomalies_df.columns:
        anomalies_df[record_id_col] = None
    if entity_id_col not in anomalies_df.columns:
        anomalies_df[entity_id_col] = None

    # Build anomaly context (optional)
    if context_cols:
        available_context_cols = [c for c in context_cols if c in anomalies_df.columns]
        anomaly_context = anomalies_df[available_context_cols].to_dict(orient="records")
    else:
        anomaly_context = [{} for _ in range(len(anomalies_df))]

    anomalies = pd.DataFrame({
        "source_table": full_table,
        "record_id": anomalies_df[record_id_col].astype(str),
        "entity_id": anomalies_df[entity_id_col].astype(str),
        "anomaly_type": model_name,
        "anomaly_score": anomalies_df["anomaly_score"],
        "anomaly_context": anomaly_context
    })

    # Write anomalies
    anomalies.to_sql(
        "anomalies",
        con=engine,
        schema="dq",
        if_exists="append",
        index=False
    )

    print(f"✅ Saved {len(anomalies)} anomalies into dq.anomalies")

    # -----------------------------
    # Save summary metrics
    # -----------------------------
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

    print("📊 Saved summary metrics into dq.anomaly_metrics")

