import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
INPUT_DIR = "pipeline_data"
BEST_K = 2  # ✅ Đã đổi thành 2 theo yêu cầu

def clustering_rfm_only():
    print(f"🚀 Running K-Means (RFM Only) with K={BEST_K}...")
    
    # 1. Load dữ liệu
    try:
        # Dữ liệu đã scale (để máy học)
        df_scaled = pd.read_csv(os.path.join(INPUT_DIR, "rfm_scaled.csv"))
        # Dữ liệu gốc (để gán nhãn)
        df_clean = pd.read_csv(os.path.join(INPUT_DIR, "data_clean.csv"))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file dữ liệu. Hãy chạy File 1 trước.")
        return

    # 2. Train K-Means
    kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    df_clean['Cluster'] = clusters
    
    # ============================================================
    # ĐẶT TÊN (LABELING) CHO 2 NHÓM
    # ============================================================
    print("🏷️  Ranking & Assigning Labels...")
    
    # Tính trung bình RFM
    summary = df_clean.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Tính điểm xếp hạng (Score)
    # R thấp = Tốt (Rank cao)
    # F & M cao = Tốt (Rank cao)
    summary['Score'] = (summary['Recency'].rank(ascending=False) + 
                        summary['Frequency'].rank(ascending=True) + 
                        summary['Monetary'].rank(ascending=True))
    
    # Sắp xếp: Cụm xịn nhất lên đầu
    summary = summary.sort_values('Score', ascending=False)
    
    print("\n📊 Cluster RFM Stats (Sorted):")
    print(summary.round(2))

    # Định nghĩa tên cho 2 nhóm
    # Nhóm 1 (Index 0): Điểm cao nhất -> VIP
    # Nhóm 2 (Index 1): Điểm thấp hơn -> Khách thường
    tier_names = [
        "Diamond (High Value)",  # Nhóm Tốt
        "Standard (Low Value)"   # Nhóm Còn lại
    ]
    
    tier_mapping = {}
    for i in range(BEST_K):
        cluster_id = summary.index[i]
        tier_mapping[cluster_id] = tier_names[i]

    # Map vào DataFrame
    df_clean['Segment'] = df_clean['Cluster'].map(tier_mapping)
    
    print("\n✅ Final Segment Mapping:")
    for cid, lbl in tier_mapping.items():
        print(f"   Cluster {cid} -> {lbl}")
        
    # In phân bố
    print("\n📈 Distribution:")
    print(df_clean['Segment'].value_counts(normalize=True).map('{:.1%}'.format))

    # 3. Lưu kết quả
    output_path = os.path.join(INPUT_DIR, "data_labeled.csv")
    df_clean.to_csv(output_path, index=False)
    
    # Lưu model
    joblib.dump(kmeans, os.path.join(INPUT_DIR, "kmeans_rfm_model.joblib"))
    
    print(f"\n💾 Saved labeled data to: {output_path}")

if __name__ == "__main__":
    clustering_rfm_only()