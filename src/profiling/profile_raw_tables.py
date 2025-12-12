import pandas as pd
from ydata_profiling import ProfileReport
from sqlalchemy import create_engine
import os
from pathlib import Path

# ---------------------------------------------------
# 1. Database Configuration
#    (reuse same details you used in load_to_postgres.py)
# ---------------------------------------------------
DB_USER = "postgres"              # ← change if needed
DB_PASSWORD = "Kiptoo.4418"     # ← change
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"    # ← change

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# ---------------------------------------------------
# 2. Helper: Profile a table and export HTML
# ---------------------------------------------------
def profile_table(schema: str, table: str, output_path: str, limit: int = 100000):
    print(f"🔎 Profiling {schema}.{table} ...")

    query = f"SELECT * FROM {schema}.{table} LIMIT {limit};"
    df = pd.read_sql(query, engine)

    # Generate profiling report
    profile = ProfileReport(
        df,
        title=f"Data Profiling Report – {schema}.{table}",
        explorative=True
    )

    # Ensure folder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    profile.to_file(output_path)
    print(f"✅ Saved profile to {output_path} (rows profiled: {len(df)})")


# ---------------------------------------------------
# 3. Main: Profile both raw tables
# ---------------------------------------------------
if __name__ == "__main__":

    # 1) UK E-Commerce dataset
    profile_table(
        schema="raw",
        table="ecommerce_transactions",
        output_path="profiling/raw_ecommerce_transactions_profile.html"
    )

    # 2) Online Retail dataset
    profile_table(
        schema="raw",
        table="online_retail",
        output_path="profiling/raw_online_retail_profile.html"
    )

    print("\n🎉 Finished generating profiling reports for raw tables.")
