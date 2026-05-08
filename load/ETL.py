import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

# ----------------------------------------
# 1. CẤU HÌNH DATABASE & FILE
# ----------------------------------------
# Thay đổi password và tên DB cho đúng máy bạn
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"


# Đường dẫn đến file csv đã merge ở bước trước
CSV_IN = "/home/tuan/Documents/scl/251/ĐATH/new_code/final_merged_data.csv"

WRITE_TO_DB = True
SCHEMA = "public"

# Kết nối DB
engine = create_engine(DB_URI)

# ======================================================
# Utilities
# ======================================================

def ensure_cols(df):
    # Các cột bắt buộc phải có trong file final_merged_data.csv
    expected = [
        'transaction_id', 'date', 'customer_id', 
        'age', 'gender', 'country', 
        'category', 'quantity', 'unit_price', 'total_amount'
    ]
    for c in expected:
        if c not in df.columns:
            print(f"⚠️ Missing column {c}, filling with NA")
            df[c] = pd.NA
    return df

def parse_dates(df):
    # Chuyển cột date sang datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# ======================================================
# 2. DIMENSION BUILDERS (Xây dựng các bảng chiều)
# ======================================================

def build_dim_date(df):
    """Tạo bảng chiều thời gian"""
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
    
    # Tạo Surrogate Key (Khóa đại diện)
    dim_date.insert(0, 'date_id', np.arange(1, len(dim_date)+1))
    return dim_date

def build_dim_category(df):
    """Tạo bảng chiều danh mục sản phẩm (Thay cho Product vì ta đã bỏ product_name)"""
    # Lấy danh sách category duy nhất
    dc = df[['category']].drop_duplicates().dropna().reset_index(drop=True)
    dc = dc.sort_values('category')
    
    # Tạo ID tự tăng
    dc.insert(0, 'category_id', np.arange(1, len(dc)+1))
    return dc

def build_dim_location(df):
    """Tạo bảng chiều địa lý"""
    dl = df[['country']].drop_duplicates().dropna().reset_index(drop=True)
    dl = dl.sort_values('country')
    
    dl.insert(0, 'location_id', np.arange(1, len(dl)+1))
    return dl

def build_dim_customer(df, dim_location):
    """Tạo bảng chiều khách hàng"""
    # Lấy thông tin khách hàng duy nhất
    dc = df[['customer_id', 'gender', 'age', 'country']].drop_duplicates(subset=['customer_id']).copy()
    
    # Merge với Location để lấy location_id thay vì lưu tên country text
    dc = dc.merge(dim_location[['country', 'location_id']], on='country', how='left')
    
    # Xóa cột country text đi cho nhẹ (vì đã có location_id)
    dc = dc.drop(columns=['country'])
    
    # Tạo Customer Key (Surrogate Key) để quản lý trong kho
    dc.insert(0, 'customer_key', np.arange(1, len(dc)+1))
    return dc

# ======================================================
# 3. FACT TABLE (Bảng sự kiện)
# ======================================================

def build_fact(df, dim_customer, dim_category, dim_date):
    """Tạo bảng Fact Sales chứa các khóa ngoại và số liệu"""
    # Copy các cột cần thiết
    f = df[['transaction_id', 'date', 'customer_id', 'category', 
            'quantity', 'unit_price', 'total_amount']].copy()

    # 1. Map Customer Key
    f = f.merge(dim_customer[['customer_id', 'customer_key']], on='customer_id', how='left')

    # 2. Map Category ID
    f = f.merge(dim_category[['category', 'category_id']], on='category', how='left')

    # 3. Map Date ID
    f['date'] = pd.to_datetime(f['date'])
    f = f.merge(dim_date[['date', 'date_id']], on='date', how='left')

    # Chọn lại các cột cuối cùng cho bảng Fact
    fact_table = f[[
        'transaction_id', 
        'date_id', 
        'customer_key', 
        'category_id', 
        'quantity', 
        'unit_price', 
        'total_amount'
    ]]

    # Tạo Primary Key cho bảng Fact (Sales ID)
    fact_table.insert(0, 'sales_id', np.arange(1, len(fact_table)+1))
    
    return fact_table

# ======================================================
# 4. PERSIST LAYER (Lưu vào DB)
# ======================================================

def persist_df(df, table_name):
    if WRITE_TO_DB:
        print(f" Saving {table_name} to DB...")
        try:
            df.to_sql(
                table_name,
                engine,
                schema=SCHEMA,
                if_exists='replace', # 'replace' để tạo mới lại bảng mỗi lần chạy
                index=False,
                method='multi',
                chunksize=5000
            )
            print(f"Wrote {len(df):,} rows to {table_name}")
        except Exception as e:
            print(f"Error writing {table_name}: {e}")
    else:
        print(f"Dry run: {table_name} prepared with {len(df)} rows.")

# ======================================================
# MAIN ETL
# ======================================================

def etl():    
    # 1. Extract
    print(" Loading CSV:", CSV_IN)
    if not os.path.exists(CSV_IN):
        print(" File not found!")
        return
    
    df = pd.read_csv(CSV_IN)
    print(f"   -> Loaded {len(df)} rows.")

    # 2. Transform & Build Dimensions
    df = ensure_cols(df)
    df = parse_dates(df)

    print("\n Building Dimensions...")
    dim_date = build_dim_date(df)
    dim_category = build_dim_category(df) # Thay cho Product
    dim_location = build_dim_location(df)
    
    # Customer cần location_id nên build sau location
    dim_customer = build_dim_customer(df, dim_location)

    print("\n Building Fact Table...")
    fact_sales = build_fact(df, dim_customer, dim_category, dim_date)

    # 3. Load (Persist)
    print("\n Loading to PostgreSQL...")
    persist_df(dim_location, "dim_location")
    persist_df(dim_category, "dim_category")
    persist_df(dim_date, "dim_date")
    persist_df(dim_customer, "dim_customer")
    persist_df(fact_sales, "fact_sales")

    print(f"\n{'='*30}\n ETL COMPLETED SUCCESSFULLY!\n{'='*30}")

