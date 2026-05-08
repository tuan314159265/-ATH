"""
RFM Analysis Module

This module contains all RFM (Recency, Frequency, Monetary) analysis functions
including data loading, RFM calculation, preprocessing, and transient customer analysis.
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
import argparse

warnings.filterwarnings("ignore")

# --- CONFIG ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MASTER = os.path.join(PROJECT_ROOT, 'final_merged_data.csv')
DEFAULT_INPUT_DIRS = [
    os.path.join(PROJECT_ROOT, "modeling", "pipeline_data"),
    os.path.join(PROJECT_ROOT, "outputs", "customers_by_segment"),
    os.path.join(PROJECT_ROOT, "outputs", "customers_vang_lai"),
    os.getcwd()
]


def load_data():
    """Load data from data warehouse for RFM analysis"""
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
    print("Syncing time periods (Shift 2011 -> 2024)...")
    def shift_date(x):
        if x.year < 2015:
            return x.replace(year = x.year + 13)
        return x
    df['date'] = df['date'].apply(shift_date)

    return df


def calculate_rfm(df):
    """Calculate RFM metrics for registered customers"""
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


def calculate_rfm_vanglai(df, date_col='date', customer_col='customer_id', amount_col='total_amount'):
    """
    Calculate RFM metrics for transient (vang lai) customers.

    Returns DataFrame with customer_id, Recency, Frequency, Monetary
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data")
    if customer_col not in df.columns:
        raise ValueError(f"Customer column '{customer_col}' not found in data")
    if amount_col not in df.columns:
        raise ValueError(f"Amount column '{amount_col}' not found in data")

    # ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # reference date: max date in the dataset
    ref_date = df[date_col].max()
    print(f"Reference date: {ref_date}")

    # group by customer
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (ref_date - x.max()).days,
        customer_col: 'size',  # Frequency = number of transactions
        amount_col: 'sum'      # Monetary = total amount
    }).rename(columns={
        date_col: 'Recency',
        customer_col: 'Frequency',
        amount_col: 'Monetary'
    })

    # add RFM scores (quintiles)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

    # combine scores
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # segment based on RFM score
    def segment_customer(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f >= 1 and m >= 2:
            return 'Potential Loyalists'
        elif r >= 2 and f >= 2 and m >= 2:
            return 'Promising'
        elif r >= 2 and f >= 1 and m >= 1:
            return 'Needs Attention'
        elif r >= 1 and f >= 2 and m >= 2:
            return 'At Risk'
        elif r >= 1 and f >= 1 and m >= 1:
            return 'Lost'
        else:
            return 'Others'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)

    return rfm.reset_index()


def remove_outliers_rfm(df):
    """Remove outliers from RFM data using IQR method"""
    print("Removing Outliers (IQR on RFM only)...")
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
    """Preprocess RFM data with Yeo-Johnson transformation and scaling"""
    print("Transforming RFM (Yeo-Johnson + Scaling)...")

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
    """Find optimal number of clusters using silhouette score"""
    print("\nFinding Optimal K (Silhouette Score)...")
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
    print(f"Best K suggested: {best_k}")
    return best_k


def extract_vanglai(master_path=None, out_path=None, chunksize=100000):
    """Extract transient customer data (customer_id == 0) from master file"""
    master = master_path or DEFAULT_MASTER
    out = out_path or os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data', 'full_info_customers_vanglai.csv')

    if not os.path.exists(master):
        print(f"Master file not found: {master}")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)

    written = 0
    # Use chunks to avoid memory spikes
    for i, chunk in enumerate(pd.read_csv(master, chunksize=chunksize)):
        # detect id column name
        id_col = None
        for c in ('customer_id', 'CustomerID', 'id'):
            if c in chunk.columns:
                id_col = c
                break
        if id_col is None:
            print("Không tìm thấy cột ID trong file master. Mong đợi 'customer_id' hoặc 'id'.")
            return 0

        sel = chunk[chunk[id_col] == 0]
        if not sel.empty:
            # write header for first chunk, append later
            if written == 0:
                sel.to_csv(out, index=False, mode='w')
            else:
                sel.to_csv(out, index=False, mode='a', header=False)
            written += len(sel)
        print(f"Processed chunk {i+1}: found {len(sel)} rows with {id_col}==0")

    print(f"Extracted {written} transient customer records to {out}")
    return written


def resolve_master_path(candidate=None):
    """Resolve path to master data file"""
    if candidate and os.path.exists(candidate):
        return candidate
    if os.path.exists(DEFAULT_MASTER):
        return DEFAULT_MASTER
    # try candidates relative to cwd
    for d in DEFAULT_INPUT_DIRS:
        p = os.path.join(d, os.path.basename(DEFAULT_MASTER))
        if os.path.exists(p):
            return p
    return None


def find_input_files(input_dir=None, pattern='customers_*.csv'):
    """Find input files matching pattern in default directories"""
    dirs = []
    if input_dir:
        dirs.append(input_dir)
    dirs.extend(DEFAULT_INPUT_DIRS)
    seen = set()
    files = []
    for d in dirs:
        if not d:
            continue
        if not os.path.exists(d):
            continue
        candidates = glob.glob(os.path.join(d, pattern))
        for c in candidates:
            if c not in seen:
                seen.add(c)
                files.append(c)
    return files


def main():
    """Main function to run RFM analysis pipeline"""
    parser = argparse.ArgumentParser(description='RFM Analysis Pipeline')
    parser.add_argument('--action', choices=['rfm', 'extract_vanglai', 'rfm_vanglai'],
                       default='rfm', help='Action to perform')
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    args = parser.parse_args()

    if args.action == 'rfm':
        # Main RFM pipeline
        df_raw = load_data()
        df_full = calculate_rfm(df_raw)

        # Clean
        df_clean = remove_outliers_rfm(df_full)

        # Lưu bản sạch đầy đủ (để File 2 dùng gán nhãn)
        df_clean.to_csv(os.path.join(OUTPUT_DIR, "data_clean.csv"), index=False)

        # Preprocess (Chỉ RFM)
        rfm_scaled = preprocess_rfm_only(df_clean)

        # Lưu bản scaled (để File 2 dùng chạy K-Means)
        pd.DataFrame(rfm_scaled, columns=['R_scaled', 'F_scaled', 'M_scaled']).to_csv(
            os.path.join(OUTPUT_DIR, "rfm_scaled.csv"), index=False
        )

        # Find K
        find_best_k(rfm_scaled)

    elif args.action == 'extract_vanglai':
        extract_vanglai(args.input, args.output)

    elif args.action == 'rfm_vanglai':
        input_file = args.input or os.path.join(OUTPUT_DIR, 'full_info_customers_vanglai.csv')
        output_file = args.output or os.path.join(OUTPUT_DIR, 'rfm_vanglai.csv')

        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            return

        df = pd.read_csv(input_file)
        rfm_vanglai = calculate_rfm_vanglai(df)
        rfm_vanglai.to_csv(output_file, index=False)
        print(f"RFM analysis for transient customers saved to {output_file}")


if __name__ == "__main__":
    main()