import pandas as pd
import numpy as np
import os
from faker import Faker
import random

# --- CẤU HÌNH ĐƯỜNG DẪN (Quan trọng: Đọc từ transformed_data) ---
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
        # Chuyển đổi date chuẩn
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    else:
        print(f"⚠️ File not found: {file_path}")
        return None

def generate_consistent_profiles(df):
    """
    Đảm bảo tính nhất quán:
    1. Nếu Kaggle thiếu Country -> Fake Country.
    2. Nếu 1 Customer ID xuất hiện ở nhiều nguồn -> Dùng chung 1 Age/Gender/Country.
    """
    print("   🔄 Harmonizing customer profiles (Age/Gender/Country)...")
    
    # Lấy danh sách tất cả khách hàng duy nhất
    unique_ids = df['customer_id'].unique()
    
    # Gom nhóm theo ID để xem khách hàng này đã có thông tin gì chưa
    # 'first' sẽ ưu tiên lấy dữ liệu có sẵn, bỏ qua NaN
    existing_info = df.groupby('customer_id')[['gender', 'age', 'country']].first().reset_index()
    
    # --- FILL CÁC TRƯỜNG CÒN THIẾU (Đặc biệt là Country cho Kaggle) ---
    
    # 1. Fill Country (Nếu thiếu)
    missing_country = existing_info['country'].isnull() | (existing_info['country'] == 'Unknown')
    count = missing_country.sum()
    if count > 0:
        print(f"   + Faking Country for {count} customers...")
        existing_info.loc[missing_country, 'country'] = [fake.country() for _ in range(count)]

    # 2. Fill Gender (Nếu còn sót)
    missing_gender = existing_info['gender'].isnull()
    count = missing_gender.sum()
    if count > 0:
        existing_info.loc[missing_gender, 'gender'] = [random.choice(['Male', 'Female']) for _ in range(count)]

    # 3. Fill Age (Nếu còn sót)
    missing_age = existing_info['age'].isnull() | (existing_info['age'] == 0)
    count = missing_age.sum()
    if count > 0:
        existing_info.loc[missing_age, 'age'] = [random.randint(18, 80) for _ in range(count)]

    return existing_info

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
    # Note: product_name đã bị xóa ở bước trước nên merge sẽ không bị lỗi
    full_df = pd.concat([df_kaggle, df_uci, df_ord], ignore_index=True)
    
    # Loại bỏ cột source (nếu không cần phân tích nguồn gốc)
    if 'source' in full_df.columns:
        full_df = full_df.drop(columns=['source'])

    print(f"📊 Raw Merged Rows: {len(full_df)}")

    # 3. Chuẩn hóa ID
    print("🛠️ Standardizing IDs...")
    full_df['transaction_id'] = full_df['transaction_id'].astype(str)
    
    # Xử lý Customer ID: Fill 'Guest' cho khách vãng lai, bỏ số thập phân
    full_df['customer_id'] = full_df['customer_id'].fillna(0).astype(str)
    full_df['customer_id'] = full_df['customer_id'].apply(lambda x: x.split('.')[0] if '.' in x else x)

    # 4. Tái chuẩn hóa Profile (Quan trọng bước này để điền Country cho Kaggle)
    # Bước này sẽ ghi đè lại các cột Age/Gender/Country trong bảng chính 
    # bằng bảng Profile chuẩn đã được lấp đầy.
    customer_profiles = generate_consistent_profiles(full_df)
    
    # Xóa cột cũ trong bảng giao dịch để tránh trùng lặp
    cols_to_drop = ['age', 'gender', 'country']
    full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])
    
    # Merge lại profile chuẩn
    full_df = pd.merge(full_df, customer_profiles, on='customer_id', how='left')

    # 5. Tính toán lại Total Amount (đảm bảo không rỗng)
    full_df['quantity'] = pd.to_numeric(full_df['quantity'], errors='coerce').fillna(0)
    full_df['unit_price'] = pd.to_numeric(full_df['unit_price'], errors='coerce').fillna(0)
    
    if 'total_amount' not in full_df.columns or full_df['total_amount'].isnull().any():
        print("💰 Recalculating Total Amount...")
        full_df['total_amount'] = full_df['quantity'] * full_df['unit_price']

    # 6. Sắp xếp cột (Final Layout)
    desired_order = [
        'transaction_id', 'date', 'customer_id', 
        'age', 'gender', 'country', 
        'category', 'quantity', 'unit_price', 'total_amount'
    ]
    # Chỉ giữ cột nào thực sự có
    final_cols = [c for c in desired_order if c in full_df.columns]
    full_df = full_df[final_cols]

    # 7. Lưu file
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*40}")
    print(f"✅ DONE! File saved to: {OUTPUT_FILE}")
    print(f"📉 Total Rows: {len(full_df)}")
    print("🔍 Missing Value Check (Should be 0):")
    print(full_df.isnull().sum())

if __name__ == "__main__":
    merge_and_fill()