# etl_load_dw.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import os

# ----------------------------------------
# CONFIG: Kết nối đúng database mydb
# ----------------------------------------
DB_URI = os.getenv(
    "DW_DB_URI",
    "postgresql+psycopg2://tuan:123456@localhost:5432/mydb"
)

CSV_IN = "new_dataset/transform_data_filled.csv"
WRITE_TO_DB = True
SCHEMA = "public"
CHUNK_SIZE = None

engine = create_engine(DB_URI)


# ======================================================
# Utilities
# ======================================================

def ensure_cols(df):
    expected = [
        'customer_id','full_name','gender','age','phone','dob','registration_date',
        'transaction_id','invoice_no','invoice_date','date','quantity',
        'unit_price','total_amount','price','category','description',
        'stock_code','country'
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def parse_dates(df):
    for c in ['date','invoice_date','dob','registration_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


# ======================================================
# Dimension builders
# ======================================================

def build_dim_date(df):
    dates = pd.to_datetime(df['date'].dropna().unique())
    dim_date = pd.DataFrame({'date': dates})
    
    dim_date['date_key'] = dim_date['date'].dt.strftime('%Y%m%d').astype(int)
    dim_date['day'] = dim_date['date'].dt.day
    dim_date['month'] = dim_date['date'].dt.month
    dim_date['month_name'] = dim_date['date'].dt.month_name()
    dim_date['quarter'] = dim_date['date'].dt.quarter
    dim_date['year'] = dim_date['date'].dt.year
    dim_date['is_weekend'] = dim_date['date'].dt.weekday >= 5

    dim_date = dim_date.sort_values('date').drop_duplicates(subset=['date'])
    dim_date.reset_index(drop=True, inplace=True)

    # Surrogate key
    dim_date.insert(0, 'date_key_surrogate', np.arange(1, len(dim_date)+1))

    return dim_date


def build_dim_product(df):
    dp = df[['stock_code','description','category']].drop_duplicates(subset=['stock_code'])
    dp['stock_code'] = dp['stock_code'].astype(str)
    dp = dp.sort_values('stock_code').reset_index(drop=True)
    dp.insert(0, 'product_key', np.arange(1, len(dp)+1))
    return dp


def build_dim_location(df):
    dl = df[['country']].drop_duplicates().reset_index(drop=True)
    dl['country'] = dl['country'].astype(str)
    dl.insert(0, 'country_key', np.arange(1, len(dl)+1))
    return dl


def build_dim_customer(df, dim_location):
    dc = df[['customer_id','full_name','gender','age','dob','registration_date','phone','country']] \
            .drop_duplicates(subset=['customer_id'])
    dc['customer_id'] = dc['customer_id'].astype(str)

    dc = dc.merge(dim_location[['country', 'country_key']], on='country', how='left')

    dc.insert(0, 'customer_key', np.arange(1, len(dc)+1))
    return dc


# ======================================================
# FACT table
# ======================================================

def build_fact(df, dim_customer, dim_product, dim_date):
    f = df[['customer_id','stock_code','date','invoice_no','transaction_id',
            'quantity','unit_price','total_amount']].copy()

    f['customer_id'] = f['customer_id'].astype(str)
    f = f.merge(dim_customer[['customer_id','customer_key']], on='customer_id', how='left')

    f['stock_code'] = f['stock_code'].astype(str)
    f = f.merge(dim_product[['stock_code','product_key']], on='stock_code', how='left')

    f['date'] = pd.to_datetime(f['date'], errors='coerce')
    f = f.merge(dim_date[['date','date_key_surrogate']], on='date', how='left')

    # rename to standard key name
    f.rename(columns={'date_key_surrogate': 'date_key'}, inplace=True)

    f['total_amount'] = f['total_amount'].fillna(f['quantity'] * f['unit_price'])

    f = f.reset_index(drop=True)
    f.insert(0, 'sales_key', np.arange(1, len(f)+1))

    return f


# ======================================================
# Persist layer
# ======================================================

def persist_df(df, table_name, if_exists='replace'):
    if WRITE_TO_DB:
        df.to_sql(
            table_name,
            engine,
            schema=SCHEMA,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=10000
        )
        print(f"✅ Wrote {len(df):,} rows to {SCHEMA}.{table_name}")
    else:
        df.to_csv(f"{table_name}.csv", index=False)
        print(f"✅ Wrote CSV {table_name}.csv")


# ======================================================
# MAIN ETL
# ======================================================

def main():
    print("Loading CSV:", CSV_IN)
    df = pd.read_csv(CSV_IN, low_memory=False) if CHUNK_SIZE is None else \
         pd.concat(pd.read_csv(CSV_IN, chunksize=CHUNK_SIZE), ignore_index=True)

    print("Raw rows:", len(df))

    df = ensure_cols(df)
    df = parse_dates(df)

    print("Building DimDate...")
    dim_date = build_dim_date(df)

    print("Building DimProduct...")
    dim_product = build_dim_product(df)

    print("Building DimLocation...")
    dim_location = build_dim_location(df)

    print("Building DimCustomer...")
    dim_customer = build_dim_customer(df, dim_location)

    print("Building FactSales...")
    fact_sales = build_fact(df, dim_customer, dim_product, dim_date)

    persist_df(dim_location, "dim_location")
    persist_df(dim_date, "dim_date")
    persist_df(dim_product, "dim_product")
    persist_df(dim_customer, "dim_customer")
    persist_df(fact_sales, "fact_sales")

    print("\n✅ ETL completed.")


if __name__ == "__main__":
    main()
