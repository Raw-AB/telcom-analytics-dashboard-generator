import pyodbc
import pandas as pd


server = '###'
database = '###'
username = '###'
password = '###'

conn_str = f"""
Driver={{ODBC Driver 18 for SQL Server}};
Server=tcp:{server},1433;
Database={database};
Uid={username};
Pwd={password};
Encrypt=yes;
TrustServerCertificate=no;
Connection Timeout=30;
"""

def run_sql(query):
    # Add transaction isolation and error handling
    with pyodbc.connect(conn_str) as conn:
        conn.autocommit = False
        try:
            df = pd.read_sql(query, conn)
            conn.commit()
            return df
        except Exception as e:
            conn.rollback()
            raise ValueError(f"SQL Error: {str(e)}")

def test_connection():
    query = "SELECT TOP 5 * FROM dim_sales_channel"  
    try:
        df = run_sql(query)
        print("Connection successful. Here's sample data:")
        print(df)
    except Exception as e:
        print("Failed to connect or query. Error:")
        print(e)

#test_connection()
