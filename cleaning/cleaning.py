import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Tuple

PATH = "/home/tuan/Documents/scl/251/ĐATH/new_code/raw_data"
CLEAN_PATH = "/home/tuan/Documents/scl/251/ĐATH/new_code/clean_data"


COLUMN_MAPPING = {
    # Kaggle Dataset
    'Transaction ID': 'transaction_id',
    'Date': 'date', 
    'Customer ID': 'customer_id',
    'Gender': 'gender',
    'Age': 'age',
    'Product Category': 'category',
    'Quantity': 'quantity', 
    'Price per Unit': 'unit_price',
    'Total Amount': 'total_amount',
    
    # UCI & Ord Dataset (Tương tự nhau)
    'InvoiceNo': 'transaction_id',
    'StockCode': 'sku_code',
    'Description': 'product_name',
    'InvoiceDate': 'date',
    'UnitPrice': 'unit_price',
    'CustomerID': 'customer_id',
    'Country': 'country'
}

TRULY_CRITICAL_COLUMNS = ['quantity', 'unit_price', 'date']

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột về chuẩn chung dựa trên mapping."""
    df = df.rename(columns=COLUMN_MAPPING)
    df.columns = [col.lower() for col in df.columns]
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các ký tự lạ thành NaN."""
    missing_indicators = [
        '-9', -9, 'NULL', 'null', 'NA', 'N/A', '', ' ', '?', 'nan', 'NaN', 
        'none', 'None', 'NAN', 'Null', 'missing', 'Missing', 'undefined'
    ]
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(missing_indicators, np.nan)
    return df_clean

def handle_missing_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Xử lý missing data """
    stats = {"rows_removed": 0, "values_imputed": 0}
    print(" Processing missing data...")
    
    df_clean = clean_missing_values(df)
    before_count = len(df_clean)

    critical_cols = [col for col in TRULY_CRITICAL_COLUMNS if col in df_clean.columns]
    if critical_cols:
        df_clean = df_clean.dropna(subset=critical_cols)
        removed = before_count - len(df_clean)
        stats["rows_removed"] += removed
        if removed > 0:
            print(f"Removed {removed} rows missing critical data {critical_cols}")

    for col in df_clean.columns:
        if not df_clean[col].isnull().any():
            continue

        missing_count = df_clean[col].isnull().sum()
        
        if col == 'customer_id':
            fill_val = 0 # Hoặc 'Guest'
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f"   👤 {col}: Filled {missing_count} missing with 'Guest/0'")
            stats["values_imputed"] += missing_count
            
        elif col in ['description', 'product_name', 'category', 'country']:
            # Thông tin mô tả -> Gán 'Unknown'
            fill_val = 'Unknown'
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f" {col}: Filled {missing_count} missing with '{fill_val}'")
            stats["values_imputed"] += missing_count
            
        elif col in ['age', 'total_amount']:
            # Dữ liệu số phụ -> Gán median
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"    {col}: Filled {missing_count} missing with median ({median_val})")
            stats["values_imputed"] += missing_count

    return df_clean, stats

def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Kiểm tra và làm sạch dữ liệu bất hợp lý (Logic bán hàng)."""
    stats = {"invalid_values_corrected": 0, "rows_dropped_logic": 0}
    print(" Validating data logic...")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        nat_count = df['date'].isna().sum()
        if nat_count > 0:
            df = df.dropna(subset=['date'])
            print(f" Dropped {nat_count} rows with invalid Date format")
            stats["rows_dropped_logic"] += nat_count

    cols_to_numeric = ['quantity', 'unit_price', 'total_amount', 'age']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'unit_price' in df.columns:
        invalid_price = df['unit_price'] < 0
        cnt = invalid_price.sum()
        if cnt > 0:
            df.loc[invalid_price, 'unit_price'] = df.loc[invalid_price, 'unit_price'].abs()
            stats["invalid_values_corrected"] += cnt
            print(f"    unit_price: Fixed {cnt} negative values (converted to absolute)")

    # Rule: Quantity
    # Trong tập UCI, Quantity âm có thể là hàng trả lại (Returns). 
    # Nếu muốn giữ hàng trả lại -> Giữ nguyên.
    # Nếu chỉ muốn phân tích doanh số bán -> Lọc Quantity > 0.
    # Ở đây mình sẽ giữ nguyên nhưng in cảnh báo.
    if 'quantity' in df.columns:
        negative_qty = (df['quantity'] < 0).sum()
        if negative_qty > 0:
            print(f" Found {negative_qty} rows with negative Quantity (likely Returns). Keeping them.")

    # Rule: Age (cho Kaggle dataset)
    if 'age' in df.columns:
        # Tuổi khách hàng phải hợp lý (ví dụ 10 - 100)
        invalid_age = (df['age'] < 10) | (df['age'] > 100)
        cnt = invalid_age.sum()
        if cnt > 0:
            df.loc[invalid_age, 'age'] = df['age'].median()
            stats["invalid_values_corrected"] += cnt
            print(f"age: Fixed {cnt} outliers (replaced with median)")

    # Rule: Gender (cho Kaggle dataset)
    if 'gender' in df.columns:
        valid_genders = ['Male', 'Female']
        # Chuẩn hóa text
        df['gender'] = df['gender'].astype(str).str.title()
        invalid_gender = ~df['gender'].isin(valid_genders)
        cnt = invalid_gender.sum()
        if cnt > 0:
            df.loc[invalid_gender, 'gender'] = 'Unknown'
            print(f"   cx gender: Fixed {cnt} invalid values to 'Unknown'")

    return df, stats

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Loại bỏ các hàng trùng lặp hoàn toàn."""
    before = len(df)
    df_new = df.drop_duplicates()
    removed = before - len(df_new)
    if removed:
        print(f"  Removed {removed} duplicate rows")
    return df_new, {"duplicates_removed": removed}

def process_file(file_path: str, dataset_name: str):
    print(f"\n{'='*40}\n PROCESSING: {dataset_name}\n{'='*40}")
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
        print(f" Loaded {len(df)} rows. Columns: {list(df.columns)}")
    except Exception as e:
        print(f" Error reading csv: {e}")
        return

    # 2. Standardize Names
    df = standardize_columns(df)

    # 3. Handle Missing Data
    df, stats_missing = handle_missing_data(df)

    # 4. Validate & Logic Check
    df, stats_logic = validate_and_clean_data(df)

    # 5. Remove Duplicates
    df, stats_dedup = remove_duplicates(df)

    # 6. Save
    if not os.path.exists(CLEAN_PATH):
        os.makedirs(CLEAN_PATH)
    
    out_name = f"clean_{dataset_name.lower()}.csv"
    out_path = os.path.join(CLEAN_PATH, out_name)
    df.to_csv(out_path, index=False)
    print(f" Saved to: {out_path} (Final rows: {len(df)})")

def cleaning():
    # Danh sách file cần xử lý
    datasets = [
        ("uci_online_retail.csv", "UCI_Retail"),
        ("kaggle_retail_sales.csv", "Kaggle_Sales"),
        ("ord.csv", "ORD_API")
    ]

    for file_name, name in datasets:
        full_path = os.path.join(PATH, file_name)
        process_file(full_path, name)
