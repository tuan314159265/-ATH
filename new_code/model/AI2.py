"""
RFM scoring and customer segmentation (FROM DATABASE)
K-means clustering for customer labeling
Train Decision Tree & Random Forest to predict cluster membership
Author: Trọng Nguyễn Lê (Fixed by Claude)
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
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# ========================== CONFIG ==========================
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"

OUTPUT_DIR = "outputs_db"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_RANGE = range(2, 7)
# ============================================================


# ============================================================
# 1) Load dữ liệu TỪ DATABASE (SQL)
# ============================================================
def load_data_from_db():
    print(f"🔌 Connecting to Database: {DB_URI.split('@')[1]}")
    
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
    
    print("📥 Executing SQL Query...")
    try:
        df = pd.read_sql(query, engine)
        
        df["Date"] = pd.to_datetime(df["Date"])
        df["Total_Amount"] = pd.to_numeric(df["Total_Amount"], errors="coerce")
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        
        print(f"✔ Loaded {len(df)} rows from Database.")
        return df
        
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        raise


# ============================================================
# 2) Compute RFM với nhiều features hơn
# ============================================================
def compute_rfm(df):
    """
    Tính RFM và thêm các features phụ để training tốt hơn
    """
    ref = df["Date"].max() + timedelta(days=1)

    # Tính các metrics cơ bản
    rfm = df.groupby("Customer_ID").agg({
        "Date": [
            lambda x: (ref - x.max()).days,  # Recency
            lambda x: (x.max() - x.min()).days  # Customer lifespan
        ],
        "InvoiceNo": [
            "nunique",  # Frequency
        ],
        "Total_Amount": [
            "sum",   # Monetary
            "mean",  # Avg order value
            "std"    # Spending variance
        ]
    }).reset_index()

    # Flatten multi-level columns
    rfm.columns = ['Customer_ID', 'Recency', 'Lifespan', 
                   'Frequency', 'Monetary', 'Avg_Order', 'Spending_Std']
    
    # Fill NaN cho Spending_Std (khách chỉ mua 1 lần)
    rfm['Spending_Std'] = rfm['Spending_Std'].fillna(0)
    
    # Tính thêm Purchase Rate (đơn/ngày)
    rfm['Purchase_Rate'] = rfm.apply(
        lambda x: x['Frequency'] / (x['Lifespan'] + 1) if x['Lifespan'] > 0 else x['Frequency'],
        axis=1
    )

    # Lấy thông tin nhân khẩu học
    demo = df.sort_values("Date").groupby("Customer_ID").last().reset_index()
    
    merge_cols = ["Customer_ID"]
    if "Age" in demo.columns: merge_cols.append("Age")
    if "Gender" in demo.columns: merge_cols.append("Gender")

    rfm = rfm.merge(demo[merge_cols], on="Customer_ID", how="left")
    
    # Xử lý missing values
    rfm["Age"] = rfm["Age"].fillna(rfm["Age"].median())
    rfm["Gender"] = rfm["Gender"].fillna("Unknown")
    
    # Encode Gender
    if rfm["Gender"].dtype == 'object':
        rfm["Gender"] = rfm["Gender"].map({'Male': 1, 'Female': 0}).fillna(-1)

    return rfm


# ============================================================
# 3) Normalize RFM thành điểm 1–5 (CHỈ để phân cụm)
# ============================================================
def rfm_quantile_scoring(rfm):
    """
    Tạo R/F/M score CHỈ để dùng cho K-means clustering
    KHÔNG dùng để train model supervised learning
    """
    df = rfm.copy()

    # Recency: Nhỏ = Tốt (5 điểm)
    df["R_score"] = pd.qcut(df["Recency"].rank(method="first"), q=5, labels=[5,4,3,2,1])

    # Frequency: Lớn = Tốt (5 điểm)
    df["F_score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])

    # Monetary: Lớn = Tốt (5 điểm)
    df["M_score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1,2,3,4,5])

    for col in ["R_score", "F_score", "M_score"]:
        df[col] = df[col].astype(int)

    return df


# ============================================================
# 4) K-means Clustering để tạo nhãn
# ============================================================
def run_kmeans(df):
    """
    Dùng K-means để phân cụm khách hàng
    Cluster này sẽ là GROUND TRUTH cho supervised learning
    """
    # Chỉ dùng RFM Score để phân cụm
    X = df[["R_score", "F_score", "M_score"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = 3
    best_sil = -1
    best_model = None

    print("\n🔍 Running KMeans to find optimal K...")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        print(f"   k={k} → silhouette={sil:.3f}")
        
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km

    print(f"✔ BEST K = {best_k}, silhouette={best_sil:.3f}")

    # Gán cluster cho mỗi khách hàng
    df["cluster"] = best_model.fit_predict(Xs)
    
    # Lưu scaler và model để dùng sau
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "kmeans_scaler.joblib"))
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, "kmeans_model.joblib"))

    return df, best_k, best_sil


# ============================================================
# 5) Phân tích cluster và tạo nhãn Loyal
# ============================================================
def map_cluster_to_loyal(rfm_clustered):
    """
    Phân tích từng cluster để xác định cluster nào là LOYAL
    Dựa trên RFM score trung bình
    """
    # Tính chỉ số trung bình của từng cluster
    cluster_analysis = rfm_clustered.groupby("cluster").agg({
        "Customer_ID": "count",
        "R_score": "mean",
        "F_score": "mean",
        "M_score": "mean",
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean"
    }).reset_index()
    
    cluster_analysis = cluster_analysis.rename(columns={"Customer_ID": "Count"})
    
    print("\n📊 Cluster Analysis:")
    print(cluster_analysis.to_string(index=False))
    
    # Tính "quality score" cho từng cluster
    # Cluster có R cao, F cao, M cao = Loyal
    cluster_analysis["quality_score"] = (
        cluster_analysis["R_score"] * 0.3 + 
        cluster_analysis["F_score"] * 0.4 + 
        cluster_analysis["M_score"] * 0.3
    )
    
    # Sắp xếp theo quality_score
    cluster_analysis = cluster_analysis.sort_values("quality_score", ascending=False)
    
    # Cluster tốt nhất = Loyal (1), các cluster khác = Non-Loyal (0)
    loyal_cluster = cluster_analysis.iloc[0]["cluster"]
    
    print(f"\n✅ Cluster {int(loyal_cluster)} được gán nhãn LOYAL")
    print(f"   - Có {cluster_analysis.iloc[0]['Count']:.0f} khách hàng")
    print(f"   - Quality Score: {cluster_analysis.iloc[0]['quality_score']:.2f}")
    
    # Tạo nhãn nhị phân từ cluster
    rfm_clustered["Cluster_Loyal"] = (
        rfm_clustered["cluster"] == loyal_cluster
    ).astype(int)
    
    # Tạo phân phối nhãn
    label_dist = rfm_clustered["Cluster_Loyal"].value_counts()
    print(f"\n📈 Label Distribution:")
    print(f"   Non-Loyal (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(rfm_clustered)*100:.1f}%)")
    print(f"   Loyal (1):     {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(rfm_clustered)*100:.1f}%)")
    
    return rfm_clustered, loyal_cluster


# ============================================================
# 6) Train Decision Tree & Random Forest
# ============================================================
def train_models(df):
    """
    Train model để dự đoán khách hàng thuộc cluster nào
    KHÔNG dùng R/F/M score làm features (tránh data leakage)
    """
    print("\n🤖 Training Models...")
    
    # ✅ Dùng giá trị thô, KHÔNG dùng score
    feature_cols = [
        "Recency",        # Số ngày kể từ lần mua cuối
        "Frequency",      # Số đơn hàng
        "Monetary",       # Tổng tiền
        "Lifespan",       # Số ngày là khách hàng
        "Avg_Order",      # Giá trị trung bình mỗi đơn
        "Purchase_Rate",  # Tần suất mua (đơn/ngày)
        "Age",
        "Gender"
    ]
    
    X = df[feature_cols]
    y = df["Cluster_Loyal"]  # ✅ Nhãn từ K-means, không phải rule-based

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Decision Tree với regularization
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=100,
        min_samples_leaf=50,
        random_state=RANDOM_STATE
    )
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=100,
        min_samples_leaf=30,
        random_state=RANDOM_STATE
    )

    # Training
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Đánh giá chi tiết
    print("\n" + "="*60)
    print("🌳 DECISION TREE PERFORMANCE")
    print("="*60)
    print(f"Train Accuracy: {dt.score(X_train, y_train):.4f}")
    print(f"Test Accuracy:  {dt.score(X_test, y_test):.4f}")
    
    y_pred_dt = dt.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dt, 
                                target_names=["Non-Loyal", "Loyal"]))
    
    # Cross-validation
    cv_scores_dt = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")

    print("\n" + "="*60)
    print("🌲 RANDOM FOREST PERFORMANCE")
    print("="*60)
    print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
    print(f"Test Accuracy:  {rf.score(X_test, y_test):.4f}")
    
    y_pred_rf = rf.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf,
                                target_names=["Non-Loyal", "Loyal"]))
    
    # Cross-validation
    cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

    # Feature importance
    print("\n📊 Feature Importance (Random Forest):")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.to_string(index=False))

    # Lưu model
    joblib.dump(dt, os.path.join(OUTPUT_DIR, "dt_classifier.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_classifier.joblib"))
    
    print(f"\n💾 Models saved to: {OUTPUT_DIR}/")

    return dt, rf


# ============================================================
# MAIN FLOW
# ============================================================
def main():
    print("="*60)
    print("🚀 RFM ANALYSIS & CUSTOMER CLUSTERING")
    print("="*60)
    
    # 1. Load Data from DB
    try:
        df = load_data_from_db()
    except Exception:
        print("⚠️ Could not load data from DB. Check connection settings.")
        return

    # 2. Tính RFM với nhiều features
    print("\n📐 Computing RFM metrics...")
    rfm = compute_rfm(df)
    
    # 3. Scoring (chỉ để K-means)
    print("📊 Creating RFM scores for clustering...")
    rfm = rfm_quantile_scoring(rfm)
    
    # 4. K-means clustering (TẠO NHÃN)
    rfm_clustered, best_k, best_sil = run_kmeans(rfm)

    # 5. Map cluster → Loyal label
    rfm_clustered, loyal_cluster = map_cluster_to_loyal(rfm_clustered)

    # 6. Train supervised learning models
    dt, rf = train_models(rfm_clustered)

    # 7. Save results
    out_path = os.path.join(OUTPUT_DIR, "rfm_results_db.csv")
    rfm_clustered.to_csv(out_path, index=False)

    print("\n" + "="*60)
    print("🎯 DONE!")
    print("="*60)
    print(f"✔ Data saved to: {out_path}")
    print(f"✔ Models saved in: {OUTPUT_DIR}/")
    print(f"✔ Best K: {best_k}, Silhouette Score: {best_sil:.3f}")
    print(f"✔ Loyal Cluster: {int(loyal_cluster)}")
    print("="*60)


if __name__ == "__main__":
    main()