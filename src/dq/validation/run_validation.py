import pandas as pd
from sqlalchemy import text
from src.config.db_config import get_engine
from src.dq.validation.rules import (
    rule_positive,
    rule_non_null,
    rule_not_future_date,
    rule_pattern,
    rule_length,
    rule_duplicates,
)


def ensure_validation_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS dq"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS dq.validation_issues (
                    source_table TEXT,
                    record_id TEXT,
                    column_name TEXT,
                    rule_name TEXT,
                    rule_description TEXT,
                    invalid_value TEXT,
                    created_at TIMESTAMP DEFAULT now()
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS dq.validation_metrics (
                    source_table TEXT,
                    rule_name TEXT,
                    total_checked INT,
                    violations INT,
                    violation_rate NUMERIC,
                    created_at TIMESTAMP DEFAULT now()
                )
                """
            )
        )


def run_validation_for_table(
    schema: str,
    table: str,
    id_col: str,
    date_col: str | None,
    numeric_cols: list,
    stock_code_col: str | None = None,
    text_cols: list | None = None,
    duplicate_keys: list[str] | None = None,
    source_label: str | None = None,
):
    """
    Apply rule-based validation to a processed table and log issues/metrics.
    """
    engine = get_engine()
    ensure_validation_tables(engine)

    full_table = f"{schema}.{table}"
    source_label = source_label or full_table

    print(f"\nRunning validation on {full_table} ...")
    df = pd.read_sql(f"SELECT * FROM {full_table}", engine)
    if df.empty:
        print(f"{full_table} is empty. Skipping validation.")
        return

    # Normalize stock_code/text columns in DB and in-memory for validation
    text_cols = text_cols or []
    existing_cols = set(df.columns)

    def _normalize_series(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )

    with engine.begin() as conn:
        if stock_code_col and stock_code_col in existing_cols:
            conn.execute(
                text(
                    f"UPDATE {full_table} SET {stock_code_col} = UPPER(REGEXP_REPLACE(TRIM({stock_code_col}), '\\s+', ' ', 'g')) WHERE {stock_code_col} IS NOT NULL"
                )
            )
            df[stock_code_col] = _normalize_series(df[stock_code_col])

        for col in text_cols:
            if col in existing_cols:
                conn.execute(
                    text(
                        f"UPDATE {full_table} SET {col} = UPPER(REGEXP_REPLACE(TRIM({col}), '\\s+', ' ', 'g')) WHERE {col} IS NOT NULL"
                    )
                )
                df[col] = _normalize_series(df[col])

    rules = []
    # ID must be present
    if id_col in df.columns:
        rules.append(lambda d: rule_non_null(d, id_col))
    # Stock code pattern/length
    if stock_code_col and stock_code_col in df.columns:
        rules.append(lambda d: rule_pattern(d, stock_code_col, r"[A-Z0-9_ ]+", f"{stock_code_col} must be alphanumeric, underscores or spaces"))
        rules.append(lambda d: rule_length(d, stock_code_col, min_len=1, max_len=20))
    # Numeric rules
    for col in numeric_cols:
        if col in df.columns:
            rules.append(lambda d, c=col: rule_positive(d, c, greater_than=0))
    # Date rule
    if date_col and date_col in df.columns:
        rules.append(lambda d: rule_not_future_date(d, date_col))
    # Duplicate rule
    if duplicate_keys:
        rules.append(lambda d, keys=duplicate_keys: rule_duplicates(d, keys))

    issues_records = []
    metrics_records = []

    for rule_fn in rules:
        mask, rule_name, rule_desc, col_name = rule_fn(df)
        if mask is None:
            continue
        violations = df[mask]
        total_checked = len(df)
        violations_count = len(violations)
        violation_rate = violations_count / total_checked if total_checked else 0

        if violations_count:
            for _, row in violations.iterrows():
                issues_records.append(
                    {
                        "source_table": source_label,
                        "record_id": str(row[id_col]) if id_col in row else None,
                        "column_name": col_name,
                        "rule_name": rule_name,
                        "rule_description": rule_desc,
                        "invalid_value": str(row.get(col_name, "")),
                    }
                )

        metrics_records.append(
            {
                "source_table": source_label,
                "rule_name": rule_name,
                "total_checked": total_checked,
                "violations": violations_count,
                "violation_rate": violation_rate,
            }
        )

    if issues_records:
        issues_df = pd.DataFrame(issues_records)
        issues_df.to_sql(
            "validation_issues",
            con=engine,
            schema="dq",
            if_exists="append",
            index=False,
        )
        print(f"Logged {len(issues_df)} validation issues for {full_table}.")
    else:
        print(f"No validation issues for {full_table}.")

    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.to_sql(
            "validation_metrics",
            con=engine,
            schema="dq",
            if_exists="append",
            index=False,
        )
        print(f"Logged validation metrics for {full_table}.")
