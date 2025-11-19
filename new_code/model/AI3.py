"""
RFM HYBRID MODEL
Combined approach:
1. Engineering: Database Connection & Pipeline Deployment (From User Code)
2. Methodology: Box-Cox, IQR Outlier Removal, Continuous K-Means (From Paper)
Author: Trọng Nguyễn Lê (Updated)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine

# Thư viện xử lý dữ liệu nâng cao (Theo bài báo)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ========================== CONFIG ==========================
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
OUTPUT_DIR = "outputs_hybrid"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_RANGE = range(2, 7)
# ============================================================

# ============================================================
# 1) LOAD DATA (Giữ nguyên ưu điểm của bạn)
# ============================================================
def load_data_from_db():
    print(f"🔌 Connecting to Database...")
    engine = create_engine(DB_URI)
    query = """
    SELECT 
        c.customer_id AS "Customer_ID",
        d.date AS "Date",
        f.transaction_id AS "InvoiceNo",
        f.total_amount AS "Total_Amount",
        c.age AS "Age",
        c.gender AS "Gender"
    FROM fact_sales f
    JOIN dim_date d ON f.date_id = d.date_id
    JOIN dim_customer c ON f.customer_key = c.customer_key
    WHERE f.total_amount > 0
    """
    try:
        df = pd.read_sql(query, engine)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Total_Amount"] = pd.to_numeric(df["Total_Amount"], errors="coerce")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0)
        print(f"✔ Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"❌ DB Error: {e}")
        raise

# ============================================================
# 2) COMPUTE RFM (Cơ bản)
# ============================================================
def compute_rfm(df):
    ref = df["Date"].max() + timedelta(days=1)
    
    rfm = df.groupby("Customer_ID").agg({
        "Date": lambda x: (ref - x.max()).days,
        "InvoiceNo": "nunique",
        "Total_Amount": "sum"
    }).reset_index()
    
    rfm.columns = ["Customer_ID", "Recency", "Frequency", "Monetary"]
    
    # Merge demographics
    demo = df.sort_values("Date").groupby("Customer_ID").last().reset_index()
    cols = ["Customer_ID"] + [c for c in ["Age", "Gender"] if c in demo.columns]
    rfm = rfm.merge(demo[cols], on="Customer_ID", how="left")
    
    # Xử lý Gender
    if "Gender" in rfm.columns and rfm["Gender"].dtype == 'object':
        rfm["Gender"] = rfm["Gender"].map({'Male': 1, 'Female': 0}).fillna(-1)
    
    return rfm

# ============================================================
# 3) ADVANCED PREPROCESSING (Theo phương pháp bài báo)
# ============================================================
def remove_outliers_iqr(df, cols=["Recency", "Frequency", "Monetary"]):
    """
    Loại bỏ ngoại lai sử dụng phương pháp IQR như tài liệu đề xuất
    Giúp cải thiện Silhouette Score
    """
    print("🧹 Removing Outliers (IQR Method)...")
    df_clean = df.copy()
    original_len = len(df_clean)
    
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        # Lọc dữ liệu nằm trong khoảng chấp nhận được
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) & 
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    
    print(f"   - Removed {original_len - len(df_clean)} outliers. Remaining: {len(df_clean)}")
    return df_clean

def preprocess_rfm_advanced(df):
    """
    1. Log/Box-Cox Transformation (Giảm độ lệch)
    2. StandardScaler (Đưa về cùng thước đo)
    """
    # Chỉ lấy R, F, M để phân cụm
    features = df[["Recency", "Frequency", "Monetary"]]
    
    # Bước 1: Power Transformation (Yeo-Johnson tốt hơn Box-Cox vì xử lý được số 0/âm)
    # Tài liệu dùng Box-Cox, ở đây dùng Yeo-Johnson tương đương nhưng an toàn hơn
    pt = PowerTransformer(method='yeo-johnson') 
    features_log = pt.fit_transform(features)
    
    # Bước 2: StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    # Lưu các bộ chuyển đổi để dùng cho deployment sau này
    joblib.dump(pt, os.path.join(OUTPUT_DIR, "power_transformer.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    
    return features_scaled, pt, scaler

# ============================================================
# 4) K-MEANS CLUSTERING (Trên dữ liệu liên tục)
# ============================================================
def run_kmeans_advanced(X_scaled):
    best_k = 3
    best_sil = -1
    best_model = None
    
    print("\n🔍 Running K-Means on Transformed Data...")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"   k={k} → silhouette={sil:.3f}")
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km
            
    print(f"✔ BEST K = {best_k}, silhouette={best_sil:.3f}")
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "kmeans_model.joblib"))
    
    return best_model.labels_, best_k

# ============================================================
# 5) DYNAMIC LABELING (Tự động định danh nhóm Loyal)
# ============================================================
def assign_loyalty_labels(df, labels):
    df["cluster"] = labels
    
    # Tính trung bình R, F, M của từng cụm (trên dữ liệu gốc)
    cluster_summary = df.groupby("cluster").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "Customer_ID": "count"
    }).reset_index()
    
    print("\n📊 Cluster Summary (Raw Data):")
    print(cluster_summary.round(2).to_string(index=False))
    
    # Logic xác định Loyal: Recency thấp nhất, Frequency & Monetary cao nhất
    # Ta tính điểm xếp hạng cho từng cụm
    # Rank Recency: Thấp là tốt (ascending=True) -> rank cao (ví dụ 1, 2, 3 -> rank 3, 2, 1)
    cluster_summary["R_rank"] = cluster_summary["Recency"].rank(ascending=False) 
    cluster_summary["F_rank"] = cluster_summary["Frequency"].rank(ascending=True)
    cluster_summary["M_rank"] = cluster_summary["Monetary"].rank(ascending=True)
    
    cluster_summary["Score"] = cluster_summary["R_rank"] + cluster_summary["F_rank"] + cluster_summary["M_rank"]
    
    # Cụm có điểm cao nhất là Loyal
    loyal_cluster_id = cluster_summary.sort_values("Score", ascending=False).iloc[0]["cluster"]
    
    print(f"\n🌟 BEST CLUSTER DETECTED: {loyal_cluster_id} (Marked as Loyal)")
    
    df["Is_Loyal"] = (df["cluster"] == loyal_cluster_id).astype(int)
    return df

# ============================================================
# 6) SUPERVISED LEARNING (Deployment)
# ============================================================
def train_classifiers(df):
    print("\n🤖 Training Classifiers (Deployment Phase)...")
    
    # Features để dự đoán (Dùng dữ liệu thô để model tự học pattern)
    features = ["Recency", "Frequency", "Monetary", "Age", "Gender"]
    X = df[features]
    y = df["Is_Loyal"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    print(f"   Random Forest Accuracy: {rf.score(X_test, y_test):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, rf.predict(X_test)))
    
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_loyal_model.joblib"))
    return rf

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("🚀 RFM HYBRID ANALYSIS (DB + Research Paper Method)")
    print("="*60)
    
    # 1. Load
    df = load_data_from_db()
    rfm_raw = compute_rfm(df)
    
    # 2. Preprocessing (Advanced: Outlier Removal -> BoxCox -> Scale)
    # Lưu ý: Ta copy ra 1 bản để phân cụm (bản này sẽ bị xóa bớt dòng do outlier)
    rfm_clean = remove_outliers_iqr(rfm_raw) 
    
    # Chuẩn hóa dữ liệu (Log/BoxCox + Scaling)
    X_scaled, pt, scaler = preprocess_rfm_advanced(rfm_clean)
    
    # 3. Clustering
    labels, best_k = run_kmeans_advanced(X_scaled)
    
    # 4. Labeling
    # Gán nhãn lại vào dataframe đã sạch
    rfm_labeled = assign_loyalty_labels(rfm_clean, labels)
    
    # 5. Deployment Model
    # Train model để dự đoán Loyal/Non-Loyal cho CẢ NHỮNG OUTLIER
    # (Vì thực tế outlier thường là khách VIP hoặc khách quá tệ, model cần học được điều này)
    # Lưu ý: Ta có thể merge nhãn về lại tập gốc nếu muốn train trên full data, 
    # nhưng train trên tập sạch thường cho core pattern tốt hơn.
    train_classifiers(rfm_labeled)
    
    # Save
    out_file = os.path.join(OUTPUT_DIR, "final_customer_segments.csv")
    rfm_labeled.to_csv(out_file, index=False)
    print(f"\n✅ Done. Results saved to: {out_file}")

if __name__ == "__main__":
    main()