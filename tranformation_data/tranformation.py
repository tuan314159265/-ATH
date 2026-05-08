import pandas as pd
import numpy as np
import os
from faker import Faker
import random
from transformers import pipeline
from tqdm import tqdm
import torch

# --- CẤU HÌNH ---
CLEAN_PATH = "/home/tuan/Documents/scl/251/ĐATH/new_code/clean_data"
OUTPUT_DIR = "/home/tuan/Documents/scl/251/ĐATH/new_code/transformed_data"

# DANH SÁCH LABEL CỐ ĐỊNH
CANDIDATE_LABELS = [
    "Beauty", "Clothing", "Electronics", "Home Decor", 
    "Kitchenware", "Toys", "Books", "Office Supplies"
]

# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Khởi tạo
fake = Faker()
Faker.seed(42)
random.seed(42)

# --- CẤU HÌNH AI (Tối ưu tốc độ GPU) ---
device = 0 if torch.cuda.is_available() else -1
print(f"⏳ Loading AI Model on device: {'GPU' if device == 0 else 'CPU'}...")

classifier = pipeline("zero-shot-classification", 
                      model="valhalla/distilbart-mnli-12-1", 
                      device=device)

def load_clean_data(filename):
    path = os.path.join(CLEAN_PATH, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- HÀM FAKE THÔNG MINH ---
def fill_demographics_consistent(df):
    """Điền Age/Gender nhất quán theo Customer ID"""
    print("   👤 Ensuring consistent demographics (Age/Gender)...")
    
    unique_customers = df['customer_id'].unique()
    profiles = []
    for cust_id in unique_customers:
        profiles.append({
            'customer_id': cust_id,
            'fake_gender': random.choice(['Male', 'Female']),
            'fake_age': random.randint(18, 75)
        })
    df_profiles = pd.DataFrame(profiles)

    # Xử lý GENDER
    if 'gender' not in df.columns:
        df = pd.merge(df, df_profiles[['customer_id', 'fake_gender']], on='customer_id', how='left')
        df = df.rename(columns={'fake_gender': 'gender'})
    else:
        df = pd.merge(df, df_profiles[['customer_id', 'fake_gender']], on='customer_id', how='left')
        mask = df['gender'].isnull() | (df['gender'] == 'Unknown')
        df.loc[mask, 'gender'] = df.loc[mask, 'fake_gender']
        df = df.drop(columns=['fake_gender'])

    # Xử lý AGE
    if 'age' not in df.columns:
        df = pd.merge(df, df_profiles[['customer_id', 'fake_age']], on='customer_id', how='left')
        df = df.rename(columns={'fake_age': 'age'})
    else:
        df = pd.merge(df, df_profiles[['customer_id', 'fake_age']], on='customer_id', how='left')
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        mask = (df['age'] == 0)
        df.loc[mask, 'age'] = df.loc[mask, 'fake_age']
        df = df.drop(columns=['fake_age'])

    return df

def apply_ai_categorization_dynamic(df, candidate_labels):
    """Phân loại sản phẩm chạy Full danh sách với tốc độ cao"""
    unique_products = df['product_name'].dropna().unique().tolist()
    total_products = len(unique_products)
    
    print(f"   🤖 AI Start Classifying {total_products} unique products using labels: {candidate_labels}")
    
    categories = []
    batch_size = 128 
    
    for i in tqdm(range(0, total_products, batch_size), desc="   Processing Batches"):
        batch = unique_products[i:i+batch_size]
        try:
            results = classifier(batch, candidate_labels)
            categories.extend([res['labels'][0] for res in results])
        except Exception as e:
            categories.extend(["Other"] * len(batch))
            
    mapping = dict(zip(unique_products, categories))
    df['category'] = df['product_name'].map(mapping).fillna("Other")
    return df

# --- 1. TRANSFORM KAGGLE ---
def transform_kaggle():
    print(f"\n{'='*10} 1. Processing Kaggle {'='*10}")
    df = load_clean_data('clean_kaggle_sales.csv')
    if df is None: return None, 0

    df['customer_id'] = df['customer_id'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
    if 'total_amount' in df.columns: df = df.drop(columns=['total_amount'])
    
    if 'category' not in df.columns and 'product category' in df.columns:
        df = df.rename(columns={'product category': 'category'})

    df = fill_demographics_consistent(df)
    
    # --- DELETE PRODUCT NAME ---
    if 'product_name' in df.columns:
        df = df.drop(columns=['product_name'])
    elif 'product name' in df.columns: # Phòng trường hợp tên chưa chuẩn
        df = df.drop(columns=['product name'])

    # Lưu file
    out_path = os.path.join(OUTPUT_DIR, "transformed_kaggle.csv")
    df['source'] = 'Kaggle' 
    df.to_csv(out_path, index=False)
    print(f"✅ Saved Kaggle to: {out_path}")

    max_id = df['transaction_id'].max() if 'transaction_id' in df.columns else len(df)
    return df, max_id

# --- 2. TRANSFORM UCI ---
def transform_uci(standard_labels, start_id_offset):
    print(f"\n{'='*10} 2. Processing UCI {'='*10}")
    df = load_clean_data('clean_uci_retail.csv')
    if df is None: return None, start_id_offset

    if 'transaction_id' not in df.columns:
        if 'invoiceno' in df.columns:
            df = df.rename(columns={'invoiceno': 'transaction_id'})
        else:
            print("   ⚠️ Auto-generating Transaction IDs...")
            df['transaction_id'] = range(start_id_offset + 1, start_id_offset + 1 + len(df))
            start_id_offset = 0 

    if start_id_offset > 0:
        df['transaction_id'] = pd.to_numeric(df['transaction_id'], errors='coerce').fillna(0).astype(int) + start_id_offset

    df['customer_id'] = df['customer_id'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
    df = fill_demographics_consistent(df)

    if 'product_name' not in df.columns and 'description' in df.columns:
        df = df.rename(columns={'description': 'product_name'})
    
    # Áp dụng AI (Cần product_name để chạy)
    if standard_labels:
        df = apply_ai_categorization_dynamic(df, standard_labels)

    # --- DELETE PRODUCT NAME (Sau khi đã dùng xong cho AI) ---
    if 'product_name' in df.columns:
        df = df.drop(columns=['product_name'])

    out_path = os.path.join(OUTPUT_DIR, "transformed_uci.csv")
    df['source'] = 'UCI'
    df.to_csv(out_path, index=False)
    print(f"✅ Saved UCI to: {out_path}")

    return df['transaction_id'].max()

# --- 3. TRANSFORM ORD ---
def transform_ord(standard_labels, start_id_offset):
    print(f"\n{'='*10} 3. Processing ORD {'='*10}")
    df = load_clean_data('clean_ord_api.csv')
    if df is None: return None

    if 'sku_code' in df.columns: df = df.drop(columns=['sku_code'])
    
    if 'transaction_id' in df.columns:
        try:
            df['transaction_id'] = df['transaction_id'].astype(int) + start_id_offset
        except: pass
    
    df['customer_id'] = df['customer_id'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
    df = fill_demographics_consistent(df)

    # Áp dụng AI (Cần product_name để chạy)
    if standard_labels:
        df = apply_ai_categorization_dynamic(df, standard_labels)

    # --- DELETE PRODUCT NAME (Sau khi đã dùng xong cho AI) ---
    if 'product_name' in df.columns:
        df = df.drop(columns=['product_name'])

    out_path = os.path.join(OUTPUT_DIR, "transformed_ord.csv")
    df['source'] = 'ORD'
    df.to_csv(out_path, index=False)
    print(f"✅ Saved ORD to: {out_path}")


def main():
    # 1. Chạy Kaggle
    df_kaggle, max_kaggle_id = transform_kaggle()

    # 2. Chạy UCI (Sử dụng CANDIDATE_LABELS cố định)
    offset_uci = max_kaggle_id + 1000
    max_uci_id = transform_uci(CANDIDATE_LABELS, offset_uci)

    # 3. Chạy ORD (Sử dụng CANDIDATE_LABELS cố định)
    offset_ord = max_uci_id + 1000
    transform_ord(CANDIDATE_LABELS, offset_ord)

    print(f"\n🎉 DONE! All files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()