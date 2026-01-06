"""
Create or refresh a date dimension table in Postgres spanning the date ranges
of raw and processed tables. This helps relate raw/processed facts to a single
Date dimension in Power BI.

Usage:
    python scripts/create_date_dim.py
"""

import pathlib
import sys

from sqlalchemy import text

# Ensure repo root on sys.path
repo_root = pathlib.Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config.db_config import get_engine


SQL = text(
    """
    CREATE SCHEMA IF NOT EXISTS dim;

    DROP TABLE IF EXISTS dim.date_dim;

    WITH bounds AS (
        SELECT
            LEAST(
                COALESCE((SELECT min(date) FROM raw.ecommerce_transactions), '1900-01-01'::date),
                COALESCE((SELECT min(invoice_date) FROM raw.online_retail), '1900-01-01'::date),
                COALESCE((SELECT min(date) FROM processed.ecommerce_transactions), '1900-01-01'::date),
                COALESCE((SELECT min(invoice_date) FROM processed.online_retail), '1900-01-01'::date)
            ) AS min_date,
            GREATEST(
                COALESCE((SELECT max(date) FROM raw.ecommerce_transactions), '2100-12-31'::date),
                COALESCE((SELECT max(invoice_date) FROM raw.online_retail), '2100-12-31'::date),
                COALESCE((SELECT max(date) FROM processed.ecommerce_transactions), '2100-12-31'::date),
                COALESCE((SELECT max(invoice_date) FROM processed.online_retail), '2100-12-31'::date)
            ) AS max_date
    ),
    dates AS (
        SELECT generate_series(min_date, max_date, interval '1 day')::date AS date_day
        FROM bounds
    )
    SELECT
        date_day,
        EXTRACT(YEAR FROM date_day)::int AS year,
        EXTRACT(MONTH FROM date_day)::int AS month,
        TO_CHAR(date_day, 'Month') AS month_name,
        EXTRACT(QUARTER FROM date_day)::int AS quarter,
        EXTRACT(WEEK FROM date_day)::int AS week,
        EXTRACT(DAY FROM date_day)::int AS day,
        TO_CHAR(date_day, 'YYYY-MM') AS year_month
    INTO dim.date_dim
    FROM dates;

    CREATE INDEX ON dim.date_dim (date_day);
    """
)


def main():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(SQL)
    print("Date dimension created/refreshed in dim.date_dim")


if __name__ == "__main__":
    main()
