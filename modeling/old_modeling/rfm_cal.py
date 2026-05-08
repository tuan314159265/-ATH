import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
OUTPUT_DIR = "pipeline_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Connecting to Data Warehouse...")
    engine = create_engine(DB_URI)
    query = """
    SELECT 
        c.customer_id, f.transaction_id, f.total_amount, d.date
        , c.age, c.gender, cat.category 
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_date d ON f.date_id = d.date_id
    JOIN dim_category cat ON f.category_id = cat.category_id
    WHERE f.total_amount > 0
    """
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    
    # --- FIX DATE GAP (Quan trọng) ---
    print("⏳ Syncing time periods (Shift 2011 -> 2024)...")
    def shift_date(x):
        if x.year < 2015:
            return x.replace(year = x.year + 13)
        return x
    df['date'] = df['date'].apply(shift_date)
    
    return df

def calculate_rfm(df):
    print("📐 Calculating Pure RFM...")
    ref_date = df['date'].max() + timedelta(days=1)
    
    # Chỉ tính R, F, M
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (ref_date - x.max()).days,
        'transaction_id': 'nunique',
        'total_amount': 'sum'
    }).rename(columns={'date': 'Recency', 'transaction_id': 'Frequency', 'total_amount': 'Monetary'})
    
    # --- PHẦN PHỤ: TÍNH CATEGORY & DEMO ĐỂ DÀNH CHO BÁO CÁO (Optional) ---
    # (Lưu lại để sau này vẽ biểu đồ Heatmap, không đưa vào K-Means)
    cat_pivot = df.pivot_table(
        index='customer_id', columns='category', values='total_amount', aggfunc='sum', fill_value=0
    )
    cat_pivot = cat_pivot.div(cat_pivot.sum(axis=1), axis=0).add_prefix('Pct_')
    
    demo = df.groupby('customer_id')[['age', 'gender']].first()
    
    # Merge lại thành bộ dữ liệu đầy đủ
    df_full = rfm.join(cat_pivot).join(demo).reset_index()
    
    # Xử lý Demo sơ bộ
    if df_full['gender'].dtype == 'object':
        df_full['gender'] = df_full['gender'].map({'Male': 1, 'Female': 0}).fillna(-1)
    df_full['age'] = df_full['age'].fillna(df_full['age'].median())
    
    return df_full

def remove_outliers_rfm(df):
    print("🧹 Removing Outliers (IQR on RFM only)...")
    # Chỉ xét ngoại lai trên 3 cột R, F, M
    cols = ['Recency', 'Frequency', 'Monetary']
    df_clean = df.copy()
    
    for c in cols:
        Q1 = df_clean[c].quantile(0.05)
        Q3 = df_clean[c].quantile(0.95)
        IQR = Q3 - Q1
        # Lọc mạnh tay (1.5) để loại bỏ các đơn bán buôn khổng lồ
        df_clean = df_clean[(df_clean[c] >= Q1 - 1.5*IQR) & (df_clean[c] <= Q3 + 1.5*IQR)]
        
    print(f"   - Removed {len(df) - len(df_clean)} rows.")
    return df_clean

def preprocess_rfm_only(df):
    print("⚡ Transforming RFM (Yeo-Johnson + Scaling)...")
    
    # CHỈ LẤY 3 CỘT NÀY ĐỂ TRAIN
    features = df[['Recency', 'Frequency', 'Monetary']]
    
    # 1. Yeo-Johnson (Chuẩn hóa phân phối)
    pt = PowerTransformer(method='yeo-johnson')
    data_trans = pt.fit_transform(features)
    
    # 2. StandardScaler (Đưa về cùng tỷ lệ)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_trans)
    
    # Lưu model
    joblib.dump(pt, os.path.join(OUTPUT_DIR, 'power_transformer.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    
    return data_scaled

def find_best_k(data_scaled):
    print("\n🔍 Finding Optimal K (Silhouette Score)...")
    print("-" * 40)
    print(f"{'K':<5} | {'Silhouette':<12} | {'Inertia'}")
    print("-" * 40)
    
    results = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        results[k] = score
        print(f"{k:<5} | {score:.4f}       | {km.inertia_:.2f}")
        
    best_k = max(results, key=results.get)
    print("-" * 40)
    print(f"✅ Best K suggested: {best_k}")
    return best_k

if __name__ == "__main__":
    # 1. Load & Calc
    df_raw = load_data()
    df_full = calculate_rfm(df_raw)
    
    # 2. Clean
    df_clean = remove_outliers_rfm(df_full)
    
    # Lưu bản sạch đầy đủ (để File 2 dùng gán nhãn)
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "data_clean.csv"), index=False)
    
    # 3. Preprocess (Chỉ RFM)
    rfm_scaled = preprocess_rfm_only(df_clean)
    
    # Lưu bản scaled (để File 2 dùng chạy K-Means)
    # Lưu dưới dạng CSV cho dễ kiểm tra
    pd.DataFrame(rfm_scaled, columns=['R_scaled', 'F_scaled', 'M_scaled']).to_csv(
        os.path.join(OUTPUT_DIR, "rfm_scaled.csv"), index=False
    )
    
    # 4. Find K
    find_best_k(rfm_scaled)