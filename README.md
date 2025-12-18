# ML Data Quality Pipeline

## Overview
End-to-end pipeline for two datasets (UK ecommerce and online retail) covering:
- Ingestion (assumed preloaded in `raw` schema)
- Profiling (HTML reports in `profiling/`)
- Anomaly detection: IsolationForest (full data), DBSCAN (sampled), KNN outlier (sampled), LOF (sampled)
- Imputation: KNN imputation (sampled)
- Validation: rule-based checks (IDs, numeric > 0, stock code pattern/length, text normalization)

Schemas used:
- `raw`: source tables (`ecommerce_transactions`, `online_retail`)
- `processed`: imputed/normalized outputs
- `dq`: anomalies, metrics, validation logs
- `audit`: imputation audit log

## Pipeline entrypoint
```
py -m src.dq.anomaly.run_anomaly_detection
```
This clears anomalies/metrics, runs all detectors, imputation, and validation for both tables.

## Sampling and parameters (default)
- IsolationForest: full data, contamination=0.02, n_estimators=200
- DBSCAN: sample 50k rows, eps=0.6, min_samples=15
- KNN outlier: sample 100k rows, k=5, contamination=0.02
- LOF: sample 100k rows, contamination=0.02
- Imputation: KNN (n_neighbors=5) on full data
- Validation: enforces ID non-null, numeric >0, stock_code pattern/length (online retail), text normalization (ecommerce product_name, retail description)

Adjust these in `src/dq/anomaly/run_anomaly_detection.py` if needed.

## Pipeline order
1) `src/ingestion/load_to_postgres.py` — load raw data into Postgres (`raw` schema).
2) `src/profiling/profile_raw_tables.py` — optional profiling of raw tables.
3) `src/dq/anomaly/run_anomaly_detection.py` - runs anomalies, imputation, and validation for both datasets (uses `detectors.py`, `imputation/knn_imputation.py`, `validation/run_validation.py` and `rules.py`).

## Outputs
- `dq.anomalies` and `dq.anomaly_metrics`
- `processed.ecommerce_transactions`, `processed.online_retail`
- `audit.imputation_log`
- `dq.validation_issues`, `dq.validation_metrics`

## Power BI tables to import
- `processed.ecommerce_transactions`
- `processed.online_retail`
- `dq.anomalies`, `dq.anomaly_metrics`
- `dq.validation_issues`, `dq.validation_metrics`
- `audit.imputation_log`

## Continuous learning runner
- `scripts/retrain_reprofile.py` can be scheduled (Task Scheduler/cron) to reprofile raw tables (optional), rerun the anomaly/imputation/validation pipeline, and log deltas for anomaly/validation metrics vs the previous run into `logs/retrain.log`. Toggle `run_profiling` in `main()` as needed.

## Configuration
DB connection in `src/config/db_config.py`. Ensure schemas (`processed`, `dq`, `audit`) are writable. Validation and imputation auto-create schemas/tables if missing.

## Requirements
See `requirements.txt`:
- pandas, numpy, scikit-learn, sqlalchemy, psycopg2-binary

## Notes
- Stock codes normalized to upper, allow alphanumeric/underscore/space; length 1-20.
- Text columns normalized (trim, collapse spaces, upper) before validation.
- Profiling reports reside in `profiling/` (optional).
