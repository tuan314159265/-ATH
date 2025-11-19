"""
RFM scoring and customer segmentation (FROM DATABASE)
K-means clustering (RFM-score)
Create Expert Loyal label
Train Decision Tree & Random Forest
Author: Trọng Nguyễn Lê
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ========================== CONFIG ==========================
# Cấu hình kết nối Database (Khớp với cấu hình ETL của bạn)
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"

OUTPUT_DIR = "outputs_db"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_RANGE = range(2, 7)

def load_data_from_db():
    print(f"Connecting to Database: {DB_URI.split('@')[1]}")
    
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
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        
        # Xử lý Gender (Nếu lưu dạng chữ thì map về số nếu cần, ở đây giữ nguyên hoặc fillna)
        # Nếu mô hình cần số, ta sẽ xử lý ở bước sau hoặc tại đây
        # Ví dụ: Male=1, Female=0 (Tạm thời để nguyên để logic phía dưới xử lý)
        
        print(f"Loaded {len(df)} rows from Database.")
        return df
        
    except Exception as e:
        print(f"Error querying database: {e}")
        raise

def compute_rfm(df):
    # Tính ngày mốc (Reference date) = Ngày giao dịch cuối cùng + 1
    ref = df["Date"].max() + timedelta(days=1)

    # Group by Customer ID
    rfm = df.groupby("Customer_ID").agg({
        "Date": lambda x: (ref - x.max()).days,     # Recency (Số ngày từ lần mua cuối)
        "InvoiceNo": "nunique",                     # Frequency (Số đơn hàng duy nhất)
        "Total_Amount": "sum"                       # Monetary (Tổng tiền)
    }).reset_index()

    rfm = rfm.rename(columns={
        "Date": "Recency",
        "InvoiceNo": "Frequency",
        "Total_Amount": "Monetary"
    })

    # Lấy thông tin nhân khẩu học (Age, Gender)
    # Lấy giá trị cuối cùng (last) của khách hàng đó
    demo = df.sort_values("Date").groupby("Customer_ID").last().reset_index()
    
    merge_cols = ["Customer_ID"]
    if "Age" in demo.columns: merge_cols.append("Age")
    if "Gender" in demo.columns: merge_cols.append("Gender")

    rfm = rfm.merge(demo[merge_cols], on="Customer_ID", how="left")
    
    # Xử lý dữ liệu Age/Gender nếu còn thiếu (fillna)
    rfm["Age"] = rfm["Age"].fillna(rfm["Age"].median())
    rfm["Gender"] = rfm["Gender"].fillna("Unknown")
    
    # Encode Gender sang số để chạy Machine Learning (Male=1, Female=0, Other=-1)
    # Lưu ý: Kiểm tra dữ liệu thực tế của bạn là 'Male'/'Female' hay 1/0
    if rfm["Gender"].dtype == 'object':
        rfm["Gender"] = rfm["Gender"].map({'Male': 1, 'Female': 0}).fillna(-1)

    return rfm


# ============================================================
# 3) Normalize RFM thành điểm 1–5 (Quantile Scoring)
# ============================================================
def rfm_quantile_scoring(rfm):
    df = rfm.copy()

    # Recency: Nhỏ = Tốt (5 điểm), Lớn = Xấu (1 điểm)
    df["R_score"] = pd.qcut(df["Recency"].rank(method="first"), q=5, labels=[5,4,3,2,1])

    # Frequency: Lớn = Tốt (5 điểm)
    df["F_score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])

    # Monetary: Lớn = Tốt (5 điểm)
    df["M_score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1,2,3,4,5])

    # Convert sang kiểu số nguyên
    for col in ["R_score", "F_score", "M_score"]:
        df[col] = df[col].astype(int)

    return df


# ============================================================
# 4) Tạo nhãn Expert Loyal
# ============================================================
def rfm_segment(r, f, m):
    # Logic phân loại khách hàng dựa trên điểm RFM
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f >= 3:
        return "Loyal Customers"
    if r >= 3 and f >= 3:
        return "Potential Loyalist"
    if r >= 4 and f <= 2:
        return "Recent Customers"
    if r == 3 and f <= 2:
        return "Promising"
    if r == 2 and f >= 3:
        return "Customers Needing Attention"
    if r == 2 and f <= 2:
        return "About to Sleep"
    if r == 1 and f >= 4:
        return "At Risk"
    if r == 1 and f == 3:
        return "Can't Lose Them"
    if r == 1 and f == 2:
        return "Hibernating"
    return "Lost"

def create_expert_label(rfm):
    df = rfm.copy()
    # Tạo tên phân khúc
    df["Segment_Name"] = df.apply(lambda x: rfm_segment(x.R_score, x.F_score, x.M_score), axis=1)
    
    # Tạo nhãn Loyal (1) / Non-Loyal (0)
    loyal_groups = {"Champions", "Loyal Customers", "Potential Loyalist"}
    df["Expert_Loyal"] = df["Segment_Name"].apply(lambda x: 1 if x in loyal_groups else 0)
    
    return df


# ============================================================
# 5) K-means Clustering
# ============================================================
def run_kmeans(df):
    # Chỉ dùng RFM Score để phân cụm
    X = df[["R_score", "F_score", "M_score"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = 4
    best_sil = -1

    print("Running KMeans to find optimal K...")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"   k={k} → silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"✔ BEST K = {best_k}, silhouette={best_sil:.3f}")

    # Chạy model cuối cùng
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km_final.fit_predict(Xs)

    return df, best_k, best_sil


# ============================================================
# 6) Train Decision Tree & Random Forest
# ============================================================
def train_models(df):
    # Features bao gồm cả RFM Score và thông tin nhân khẩu học
    feature_cols = ["R_score", "F_score", "M_score", "Age", "Gender"]
    X = df[feature_cols]
    y = df["Expert_Loyal"]

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

    print("\nTraining Models...")
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    print(f"   Decision Tree Test Acc: {dt.score(X_test, y_test):.4f}")
    print(f"   Random Forest Test Acc: {rf.score(X_test, y_test):.4f}")

    # Lưu model
    joblib.dump(dt, os.path.join(OUTPUT_DIR, "dt.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf.joblib"))

    return dt, rf


# ============================================================
# MAIN FLOW
# ============================================================
def main():
    # 1. Load Data from DB
    try:
        df = load_data_from_db()
    except Exception:
        print("⚠️ Could not load data from DB. Check connection settings.")
        return

    # 2. RFM Calculation
    rfm = compute_rfm(df)
    
    # 3. Scoring
    rfm = rfm_quantile_scoring(rfm)
    
    # 4. Segmentation
    rfm = create_expert_label(rfm)

    # 5. Clustering
    rfm_clustered, best_k, best_sil = run_kmeans(rfm)

    # 6. Modeling
    dt, rf = train_models(rfm_clustered)

    # Save Result
    out_path = os.path.join(OUTPUT_DIR, "rfm_results_db.csv")
    rfm_clustered.to_csv(out_path, index=False)

    print("\n🎯 DONE.")
    print(f"   Data saved to: {out_path}")
    print(f"   Models saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()