import pandas as pd
import numpy as np
import os
from faker import Faker
import random

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đọc dữ liệu từ bước transform trước đó
TRANSFORMED_PATH = "/home/tuan/Documents/scl/251/ĐATH/new_code/transformed_data"
OUTPUT_FILE = "/home/tuan/Documents/scl/251/ĐATH/new_code/final_merged_data.csv"

fake = Faker()
Faker.seed(42)
random.seed(42)

def load_data(filename):
    file_path = os.path.join(TRANSFORMED_PATH, filename)
    if os.path.exists(file_path):
        print(f"📥 Loading {filename}...")
        df = pd.read_csv(file_path)
        
        # --- XỬ LÝ NGÀY THÁNG (CHỈ GIỮ NGÀY) ---
        if 'date' in df.columns:
            # 1. Chuyển về dạng datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # 2. Chỉ lấy phần ngày (YYYY-MM-DD), bỏ giờ
            df['date'] = df['date'].dt.date 
            
        return df
    else:
        print(f"⚠️ File not found: {file_path}")
        return None

def merge_and_fill():
    print(f"{'='*40}\n 🚀 MERGING FINAL DATASETS \n{'='*40}")

    # 1. Load Files (File đã qua xử lý AI)
    df_kaggle = load_data("transformed_kaggle.csv")
    df_uci = load_data("transformed_uci.csv")
    df_ord = load_data("transformed_ord.csv")

    if df_uci is None and df_kaggle is None and df_ord is None:
        print("❌ No data found. Please run transform script first.")
        return

    # 2. Gộp data (Concat)
    full_df = pd.concat([df_kaggle, df_uci, df_ord], ignore_index=True)
    
    # Loại bỏ cột source (nếu muốn giữ lại để biết nguồn thì xóa dòng này đi)
    if 'source' in full_df.columns:
        full_df = full_df.drop(columns=['source'])

    print(f"📊 Raw Merged Rows: {len(full_df)}")

    # 3. Chuẩn hóa ID
    print("🛠️ Standardizing IDs...")
    full_df['transaction_id'] = full_df['transaction_id'].astype(str)
    
    full_df['customer_id'] = full_df['customer_id'].fillna(0).astype(str)
    full_df['customer_id'] = full_df['customer_id'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    
    # Xóa cột cũ trong bảng giao dịch để tránh trùng lặp
    cols_to_drop = ['age', 'gender', 'country']
    full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])
    
    # 5. Tính toán lại Total Amount
    full_df['quantity'] = pd.to_numeric(full_df['quantity'], errors='coerce').fillna(0)
    full_df['unit_price'] = pd.to_numeric(full_df['unit_price'], errors='coerce').fillna(0)
    
    if 'total_amount' not in full_df.columns or full_df['total_amount'].isnull().any():
        print("Recalculating Total Amount...")
        full_df['total_amount'] = full_df['quantity'] * full_df['unit_price']

    # 6. Sắp xếp cột (Final Layout)
    desired_order = [
        'transaction_id', 'date', 'customer_id', 
        'age', 'gender', 'country', 
        'category', 'quantity', 'unit_price', 'total_amount'
    ]
    final_cols = [c for c in desired_order if c in full_df.columns]
    full_df = full_df[final_cols]

    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*40}")
    print(f"File saved to: {OUTPUT_FILE}")
    print(f"Total Rows: {len(full_df)}")
    
    print("\nSample Data (Check Date Format):")
    print(full_df[['date']].head())

if __name__ == "__main__":
    merge_and_fill()