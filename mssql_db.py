# mssql_db.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ví dụ: SQL auth
SQL_SERVER   = os.getenv("SQL_SERVER", r"localhost\SQLEXPRESS")
SQL_DB       = os.getenv("SQL_DB", "MusicApp")
SQL_USER     = os.getenv("SQL_USER", "sa")          # nếu dùng SQL auth
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "YourPass")# nếu dùng SQL auth

# Dùng SQL Auth:
SQLALCHEMY_URL = (
    f"mssql+pyodbc://{SQL_USER}:{SQL_PASSWORD}@{SQL_SERVER}/{SQL_DB}"
    "?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes"
)

# Nếu muốn Windows Auth (Trusted_Connection):
# SQLALCHEMY_URL = (
#   f"mssql+pyodbc://@{SQL_SERVER}/{SQL_DB}"
#   "?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&Encrypt=yes&TrustServerCertificate=yes"
# )

engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True, fast_executemany=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def fetch_all(query, params=None):
    with engine.connect() as conn:
        rows = conn.execute(text(query), params or {}).mappings().all()
        return [dict(r) for r in rows]

def execute(query, params=None):
    with engine.begin() as conn:
        conn.execute(text(query), params or {})
