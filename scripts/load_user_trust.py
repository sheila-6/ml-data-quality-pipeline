"""
Load user trust scores from a CSV into trust.user_trust.
Expected CSV columns:
- source_table (e.g., processed.ecommerce_transactions)
- user_group (e.g., analyst, manager)
- trust_score_raw (1-5 Likert)

We normalize trust_score_raw to 0-1 as (score-1)/4 and average per source_table/user_group.

Usage:
    python scripts/load_user_trust.py path/to/trust_scores.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text
from src.config.db_config import get_engine


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/load_user_trust.py path/to/trust_scores.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    agg = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_table = row.get("source_table")
            user_group = row.get("user_group")
            raw = row.get("trust_score_raw")
            if not source_table or not user_group or raw is None:
                continue
            try:
                raw_val = float(raw)
            except ValueError:
                continue
            norm = max(0.0, min(1.0, (raw_val - 1.0) / 4.0))
            agg[(source_table, user_group)].append(norm)

    averaged = []
    for (source_table, user_group), vals in agg.items():
        if not vals:
            continue
        averaged.append(
            {
                "source_table": source_table,
                "user_group": user_group,
                "trust_score": sum(vals) / len(vals),
            }
        )

    engine = get_engine()
    ddl = text(
        """
        CREATE SCHEMA IF NOT EXISTS trust;
        CREATE TABLE IF NOT EXISTS trust.user_trust (
            source_table TEXT NOT NULL,
            user_group   TEXT NOT NULL,
            trust_score  NUMERIC CHECK (trust_score >= 0 AND trust_score <= 1),
            collected_at TIMESTAMP DEFAULT now()
        );
        DELETE FROM trust.user_trust;
        """
    )
    with engine.begin() as conn:
        conn.execute(ddl)
        for row in averaged:
            conn.execute(
                text(
                    """
                    INSERT INTO trust.user_trust (source_table, user_group, trust_score)
                    VALUES (:source_table, :user_group, :trust_score)
                    """
                ),
                row,
            )

    print(f"Loaded {len(averaged)} trust scores into trust.user_trust")


if __name__ == "__main__":
    main()
