import pandas as pd
from sqlalchemy import create_engine
import os

# ---------------------------------------------------
# 1. Database Configuration
# ---------------------------------------------------

DB_USER = "postgres"              # ← change if needed
DB_PASSWORD = "Kiptoo.4418"     # ← change this
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"    # ← change this

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# ---------------------------------------------------
# 2. Load UK E-Commerce Dataset (Dataset 1)
#    Columns: TransactionNo, Date, ProductNo, ProductName, Price, Quantity, CustomerNo, Country
# ---------------------------------------------------

def load_ecommerce_dataset(file_path: str):
    print("🔹 Loading UK E-Commerce dataset...")
    df = pd.read_csv(file_path)

    # Make sure column order matches your file
    # Adjust if your CSV has different names
    df.columns = [
        "transaction_no",
        "date",
        "product_no",
        "product_name",
        "price",
        "quantity",
        "customer_no",
        "country"
    ]

    # Convert identifiers to string (very important!)
    df["transaction_no"] = df["transaction_no"].astype(str)
    df["product_no"] = df["product_no"].astype(str)
    df["customer_no"] = df["customer_no"].astype(str)

    # Parse date (no time-of-day in this dataset)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Optional: ensure numeric types where appropriate
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")

    # Write to Postgres (raw.ecommerce_transactions)
    df.to_sql(
        "ecommerce_transactions",
        con=engine,
        schema="raw",
        if_exists="append",   # or "replace" during testing
        index=False
    )

    print(f"✅ Loaded {len(df)} rows into raw.ecommerce_transactions")


# ---------------------------------------------------
# 3. Load Online Retail Dataset (Dataset 2)
#    Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
# ---------------------------------------------------

def load_online_retail_dataset(file_path: str):
    print("🔹 Loading Online Retail (UCI) dataset...")

    # Usually this is an Excel file from UCI
    if file_path.lower().endswith(".xlsx") or file_path.lower().endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df.columns = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country"
    ]

    # Convert identifiers to string (to avoid bigint errors)
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    df["customer_id"] = df["customer_id"].astype(str)

    # Parse timestamp with time-of-day
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    # Numeric fields
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")

    # Write to Postgres (raw.online_retail)
    df.to_sql(
        "online_retail",
        con=engine,
        schema="raw",
        if_exists="append",   # or "replace" during testing
        index=False
    )

    print(f"✅ Loaded {len(df)} rows into raw.online_retail")


# ---------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------

if __name__ == "__main__":
    # 🔁 TODO: set your actual file paths here
    ecommerce_path = r"C:\Users\Admin\Desktop\Master Thesis\ml-data-quality-pipeline\data\raw\uk_ecommerce.csv"
    online_retail_path = r"C:\Users\Admin\Desktop\Master Thesis\ml-data-quality-pipeline\data\raw\online_retail.xlsx"

    load_ecommerce_dataset(ecommerce_path)
    load_online_retail_dataset(online_retail_path)


    print("\n🎉 All datasets successfully loaded into PostgreSQL!")