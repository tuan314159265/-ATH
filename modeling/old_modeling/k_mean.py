import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
INPUT_DIR = "pipeline_data"
BEST_K = 2  # Số nhóm cho K-Means

def clustering_and_split_id_only():
    print(f"🚀 Running K-Means & Splitting IDs (K={BEST_K})...")
    
    # 1. Load dữ liệu
    try:
        df_scaled = pd.read_csv(os.path.join(INPUT_DIR, "rfm_scaled.csv"))
        df_clean = pd.read_csv(os.path.join(INPUT_DIR, "data_clean.csv"))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file dữ liệu.")
        return

    # 2. Train K-Means
    kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    df_clean['Cluster'] = clusters
    
    # 3. Đặt tên nhãn (Diamond/Standard)
    print("🏷️  Ranking & Assigning Labels...")
    summary = df_clean.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    summary['Score'] = (summary['Recency'].rank(ascending=False) + 
                        summary['Frequency'].rank(ascending=True) + 
                        summary['Monetary'].rank(ascending=True))
    summary = summary.sort_values('Score', ascending=False)
    
    tier_names = ["Diamond", "Standard"]
    tier_mapping = {}
    for i in range(BEST_K):
        tier_mapping[summary.index[i]] = tier_names[i]

    df_clean['Segment'] = df_clean['Cluster'].map(tier_mapping)

    # 4. Xác định tên cột ID
    # Code sẽ tự tìm xem trong file csv cột ID tên là gì
    id_col = 'customer_id'
    if id_col not in df_clean.columns:
        if 'id' in df_clean.columns:
            id_col = 'id'
        else:
            print("❌ Lỗi: Không tìm thấy cột CustomerID hoặc id.")
            return

    # 5. Tách DataFrame
    # Tách Vãng lai (ID = 0) và Khách có đăng ký (ID != 0)
    df_vanglai = df_clean[df_clean[id_col] == 0]
    df_registered = df_clean[df_clean[id_col] != 0]

    # Tách Diamond và Standard từ nhóm đã đăng ký
    df_diamond = df_registered[df_registered['Segment'] == 'Diamond']
    df_standard = df_registered[df_registered['Segment'] == 'Standard']

    # ============================================================
    # LƯU FILE (CHỈ GIỮ LẠI CỘT ID)
    # ============================================================``
    print("\n📂 Saving ID-only files...")
    
    path_vanglai = os.path.join(INPUT_DIR, "customers_vanglai.csv")
    path_diamond = os.path.join(INPUT_DIR, "customers_diamond.csv")
    path_standard = os.path.join(INPUT_DIR, "customers_standard.csv")

    # Cú pháp [[id_col]] giúp chỉ chọn cột đó để lưu
    df_vanglai[[id_col]].to_csv(path_vanglai, index=False)
    df_diamond[[id_col]].to_csv(path_diamond, index=False)
    df_standard[[id_col]].to_csv(path_standard, index=False)
    
    # Lưu model K-means
    joblib.dump(kmeans, os.path.join(INPUT_DIR, "kmeans_rfm_model.joblib"))

    print("\n📊 Export Summary (ID columns only):")
    print(f"   1. Vãng lai (ID=0) : {len(df_vanglai)} IDs -> {path_vanglai}")
    print(f"   2. Diamond (VIP)   : {len(df_diamond)} IDs -> {path_diamond}")
    print(f"   3. Standard        : {len(df_standard)} IDs -> {path_standard}")

if __name__ == "__main__":
    clustering_and_split_id_only()