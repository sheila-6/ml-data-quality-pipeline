"""
Lightweight continuous-learning runner:
- Reprofile raw tables (optional skip)
- Rerun anomaly pipeline (includes LOF/DBSCAN/KNN/IsolationForest)
- Compare anomaly/validation metrics to prior run and log deltas

Run on a schedule (Task Scheduler/cron) to detect drift and keep metrics fresh.
"""

import datetime
import logging
from pathlib import Path

from sqlalchemy import text
from src.config.db_config import get_engine
from src.dq.anomaly.run_anomaly_detection import setup_logging, run_for_ecommerce, run_for_online_retail
from src.profiling.profile_raw_tables import profile_table


def fetch_latest_metrics():
    engine = get_engine()
    with engine.connect() as conn:
        anomaly = conn.execute(
            text(
                """
                select source_table, model_name, anomaly_rate, created_at
                from dq.anomaly_metrics
                order by created_at desc
                """
            )
        ).mappings().all()
        validation = conn.execute(
            text(
                """
                select source_table, rule_name, violation_rate, created_at
                from dq.validation_metrics
                order by created_at desc
                """
            )
        ).mappings().all()
    return anomaly, validation


def compare_metrics(current, previous, key_fields):
    prev_map = {(tuple(item[k] for k in key_fields)): item for item in previous}
    deltas = []
    for item in current:
        key = tuple(item[k] for k in key_fields)
        prev = prev_map.get(key)
        if prev:
            delta = {}
            for k, v in item.items():
                if k in key_fields:
                    delta[k] = v
                elif isinstance(v, (int, float)):
                    delta[k] = v - (prev.get(k) or 0)
            deltas.append(delta)
    return deltas


def main(run_profiling: bool = False):
    logger = setup_logging("logs/retrain.log")
    logger.info("Starting retrain/reprofile cycle at %s", datetime.datetime.utcnow())

    if run_profiling:
        Path("profiling").mkdir(exist_ok=True)
        logger.info("Reprofiling raw tables (sampled)...")
        profile_table("raw", "ecommerce_transactions", "profiling/raw_ecommerce_transactions_profile.html")
        profile_table("raw", "online_retail", "profiling/raw_online_retail_profile.html")

    prev_anomaly, prev_validation = fetch_latest_metrics()

    # Rerun pipeline for both datasets
    logger.info("Running pipeline for ecommerce...")
    run_for_ecommerce()
    logger.info("Running pipeline for online_retail...")
    run_for_online_retail()

    curr_anomaly, curr_validation = fetch_latest_metrics()

    anomaly_deltas = compare_metrics(curr_anomaly, prev_anomaly, ["source_table", "model_name"])
    validation_deltas = compare_metrics(curr_validation, prev_validation, ["source_table", "rule_name"])

    # Warn on significant drift
    anomaly_thresh = 0.02  # 2 percentage points
    violation_thresh = 0.02
    if anomaly_deltas:
        logger.info("Anomaly metric deltas vs prior run: %s", anomaly_deltas)
        for delta in anomaly_deltas:
            dr = delta.get("anomaly_rate")
            if dr is not None and abs(dr) > anomaly_thresh:
                logger.warning("Anomaly rate drift > %.2f: %s", anomaly_thresh, delta)
    if validation_deltas:
        logger.info("Validation metric deltas vs prior run: %s", validation_deltas)
        for delta in validation_deltas:
            dr = delta.get("violation_rate")
            if dr is not None and abs(dr) > violation_thresh:
                logger.warning("Validation violation_rate drift > %.2f: %s", violation_thresh, delta)

    logger.info("Retrain/reprofile cycle completed.")


if __name__ == "__main__":
    main(run_profiling=False)
