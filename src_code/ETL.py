# etl_load_dw.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import os

# ---------- CONFIG ----------
DB_URI = os.getenv("DW_DB_URI", "postgresql://user:password@localhost:5432/dwdb")
CSV_IN = "transform_data.csv"
WRITE_TO_DB = True   # False -> write CSVs instead of DB
SCHEMA = "public"
CHUNK_SIZE = None    # None -> load whole file; or set e.g. 200000 for big files
# ----------------------------

engine = create_engine(DB_URI)

def ensure_cols(df):
    # Basic normalisation & required columns
    expected = ['customer_id','full_name','gender','age','phone','dob','registration_date',
                'transaction_id','invoice_no','invoice_date','date',
                'quantity','unit_price','total_amount','price','category','description',
                'stock_code','country']
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def parse_dates(df):
    # Try parse common date columns
    for c in ['date','invoice_date','dob','registration_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

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
    dc = df[['customer_id','full_name','gender','age','dob','registration_date','phone','country']].drop_duplicates(subset=['customer_id'])
    dc['customer_id'] = dc['customer_id'].astype(str)
    # Map country -> country_key
    dc = dc.merge(dim_location[['country','country_key']], on='country', how='left')
    dc.insert(0, 'customer_key', np.arange(1, len(dc)+1))
    return dc

def build_fact(df, dim_customer, dim_product, dim_date):
    f = df[['customer_id','stock_code','date','invoice_no','transaction_id','quantity','unit_price','total_amount']]
    # map customer_key
    f['customer_id'] = f['customer_id'].astype(str)
    f = f.merge(dim_customer[['customer_id','customer_key']], on='customer_id', how='left')
    f['stock_code'] = f['stock_code'].astype(str)
    f = f.merge(dim_product[['stock_code','product_key']], on='stock_code', how='left')
    # map date_key
    f['date'] = pd.to_datetime(f['date'], errors='coerce')
    f = f.merge(dim_date[['date','date_key']], left_on='date', right_on='date', how='left')
    # rename date_key -> date_key_surrogate if needed
    if 'date_key_surrogate' in dim_date.columns:
        f = f.merge(dim_date[['date','date_key_surrogate']], on='date', how='left')
        f.rename(columns={'date_key_surrogate':'date_key'}, inplace=True)
    # Fill total_amount if missing
    f['total_amount'] = f['total_amount'].fillna(f['quantity'] * f['unit_price'])
    # sales surrogate
    f = f.reset_index(drop=True)
    f.insert(0, 'sales_key', np.arange(1, len(f)+1))
    # final ordering and column selection
    f = f[['sales_key','customer_key','product_key','date_key','invoice_no','transaction_id','quantity','unit_price','total_amount']]
    return f

def persist_df(df, table_name, if_exists='replace'):
    if WRITE_TO_DB:
        df.to_sql(table_name, engine, schema=SCHEMA, if_exists=if_exists, index=False, method='multi', chunksize=10000)
        print(f"Wrote {len(df):,} rows to {table_name}")
    else:
        fn = f"{table_name}.csv"
        df.to_csv(fn, index=False)
        print(f"Wrote CSV {fn}")

def main():
    print("Loading CSV:", CSV_IN)
    if CHUNK_SIZE is None:
        df = pd.read_csv(CSV_IN, low_memory=False)
    else:
        # Read in chunks and concat (useful for huge files)
        chunks = []
        for chunk in pd.read_csv(CSV_IN, chunksize=CHUNK_SIZE, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    print("Raw rows:", len(df))

    df = ensure_cols(df)
    df = parse_dates(df)

    # Build dims
    print("Building DimDate...")
    dim_date = build_dim_date(df)
    # create a simple date_key col for joins on integer YYYYMMDD
    dim_date = dim_date.rename(columns={'date_key_surrogate':'date_key'})

    print("Building DimProduct...")
    dim_product = build_dim_product(df)

    print("Building DimLocation...")
    dim_location = build_dim_location(df)

    print("Building DimCustomer...")
    dim_customer = build_dim_customer(df, dim_location)

    print("Building FactSales...")
    fact_sales = build_fact(df, dim_customer, dim_product, dim_date)

    # Persist to DB / CSV
    persist_df(dim_location, 'dim_location')
    persist_df(dim_date[['date_key','date','day','month','month_name','quarter','year','is_weekend']], 'dim_date')
    persist_df(dim_product, 'dim_product')
    persist_df(dim_customer, 'dim_customer')
    persist_df(fact_sales, 'fact_sales')

    print("ETL completed.")

if __name__ == "__main__":
    main()
