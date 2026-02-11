import os
from sqlalchemy import create_engine

# Database configuration
DB_USER = "postgres"         
DB_PASSWORD = "xxxxx"  # Replace with your actual password
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres" 

def get_engine():
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    return engine
