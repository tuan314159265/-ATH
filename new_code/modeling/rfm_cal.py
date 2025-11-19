import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
OUTPUT_DIR = "pipeline_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    print("Connecting to Data Warehouse")
    engine = create_engine(DB_URI)
    query = """
    SELECT 
        c.customer_id, c.age, c.gender, 
        f.transaction_id, f.total_amount, d.date, cat.category
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_date d ON f.date_id = d.date_id
    JOIN dim_category cat ON f.category_id = cat.category_id
    WHERE f.total_amount > 0
    """
    return pd.read_sql(query, engine)

def feature_engineering(df):
    print("Feature Engineering (RFM + Category + Demographics)...")
    
    # 1. Tính RFM
    ref_date = pd.to_datetime(df['date']).max() + timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (ref_date - pd.to_datetime(x).max()).days,
        'transaction_id': 'nunique',
        'total_amount': 'sum'
    }).rename(columns={'date': 'Recency', 'transaction_id': 'Frequency', 'total_amount': 'Monetary'})
    
    # 2. Tính tỷ lệ Category Preference (%)
    cat_pivot = df.pivot_table(
        index='customer_id', columns='category', values='total_amount', aggfunc='sum', fill_value=0
    )
    cat_pivot = cat_pivot.div(cat_pivot.sum(axis=1), axis=0).add_prefix('Pct_')
    
    # 3. Lấy Demographics
    demo = df.groupby('customer_id')[['age', 'gender']].first()
    
    # Merge tất cả
    data_final = rfm.join(cat_pivot).join(demo).reset_index()
    
    # Xử lý Gender (Male/Female -> 1/0) & Age
    data_final['gender'] = data_final['gender'].map({'Male': 1, 'Female': 0}).fillna(-1)
    data_final['age'] = data_final['age'].fillna(data_final['age'].median())
    
    return data_final

def remove_outliers_iqr(df):
    # Chỉ lọc trên RFM, giữ lại Category/Demo
    print("Removing Outliers (IQR Method based on Research Paper)")
    cols = ['Recency', 'Frequency', 'Monetary']
    df_clean = df.copy()
    
    for c in cols:
        Q1 = df_clean[c].quantile(0.05)
        Q3 = df_clean[c].quantile(0.95)
        IQR = Q3 - Q1
        df_clean = df_clean[(df_clean[c] >= Q1 - 1.5*IQR) & (df_clean[c] <= Q3 + 1.5*IQR)]
        
    print(f"   - Removed {len(df) - len(df_clean)} rows.")
    return df_clean

def preprocess_pipeline(df):
    print("Transforming & Scaling...")
    
    # Tách feature ID ra
    ids = df[['customer_id']]
    features = df.drop(columns=['customer_id'])
    
    # Power Transform cho các cột bị lệch (RFM)
    skewed_cols = ['Recency', 'Frequency', 'Monetary']
    other_cols = [c for c in features.columns if c not in skewed_cols]
    
    # Pipeline xử lý
    # 1. Yeo-Johnson cho RFM (giảm lệch)
    # 2. StandardScaler cho tất cả (đưa về cùng hệ quy chiếu)
    preprocessor = ColumnTransformer(
        transformers=[
            ('power', PowerTransformer(method='yeo-johnson'), skewed_cols),
            ('passthrough', 'passthrough', other_cols) # Giữ nguyên các cột khác để scale sau
        ]
    )
    
    # Fit & Transform
    data_transformed = preprocessor.fit_transform(features)
    
    # Scale toàn bộ
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_transformed)
    
    # (Optional) PCA - Giảm chiều nếu có quá nhiều Category
    pca = PCA(n_components=0.95) # Giữ lại 95% thông tin
    data_pca = pca.fit_transform(data_scaled)
    print(f"   - PCA reduced features from {data_scaled.shape[1]} to {data_pca.shape[1]}")
    
    # Lưu preprocessors
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    joblib.dump(pca, os.path.join(OUTPUT_DIR, 'pca.joblib'))
    
    return data_pca, df

def find_k(data_pca):
    print("\nFinding Optimal K (Silhouette Score)...")
    results = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(data_pca)
        score = silhouette_score(data_pca, labels)
        results[k] = score
        print(f"   K={k}: Silhouette={score:.4f}")
        
    best_k = max(results, key=results.get)
    print(f"Best K based on Silhouette: {best_k}")
    return best_k

def rfm_cal():
    df_raw = load_data()
    df_features = feature_engineering(df_raw)
    df_clean = remove_outliers_iqr(df_features)
    
    # Lưu dữ liệu sạch chưa scale để dùng cho File 2
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "data_clean.csv"), index=False)
    
    # Xử lý và tìm K
    data_ready, _ = preprocess_pipeline(df_clean)
    find_k(data_ready)
    
    # Lưu dữ liệu đã PCA để dùng cho clustering nhanh
    np.save(os.path.join(OUTPUT_DIR, "data_pca.npy"), data_ready)