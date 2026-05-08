"""
Clustering Module

This module contains all clustering algorithms and customer segmentation functions,
primarily K-means clustering for RFM data.
"""

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import joblib
import warnings
import argparse

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
INPUT_DIR = "data"
BEST_K = 2  # Số nhóm cho K-Means


def clustering_and_split_id_only(best_k=None):
    """Perform K-means clustering and split customers by segment"""
    k = best_k or BEST_K
    print(f"Running K-Means & Splitting IDs (K={k})...")

    # 1. Load dữ liệu
    try:
        df_scaled = pd.read_csv(os.path.join(INPUT_DIR, "rfm_scaled.csv"))
        df_clean = pd.read_csv(os.path.join(INPUT_DIR, "data_clean.csv"))
    except FileNotFoundError:
        print("[ERROR] Required files not found in pipeline_data/. Please run RFM analysis first.")
        return

    # 2. Train K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    df_clean['Cluster'] = clusters

    # 3. Đặt tên nhãn (Diamond/Standard)
    print("Ranking & Assigning Labels...")
    summary = df_clean.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    summary['Score'] = (summary['Recency'].rank(ascending=False) +
                        summary['Frequency'].rank(ascending=True) +
                        summary['Monetary'].rank(ascending=True))
    summary = summary.sort_values('Score', ascending=False)

    tier_names = ["Diamond", "Standard"]
    tier_mapping = {}
    for i in range(k):
        tier_mapping[summary.index[i]] = tier_names[i]

    df_clean['Segment'] = df_clean['Cluster'].map(tier_mapping)

    # 4. Xác định tên cột ID
    # Code sẽ tự tìm xem trong file csv cột ID tên là gì
    id_col = 'customer_id'
    if id_col not in df_clean.columns:
        if 'id' in df_clean.columns:
            id_col = 'id'
        else:
            print("[ERROR] No ID column found in data.")
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
    # ============================================================
    print("\n Saving ID-only files...")

    path_vanglai = os.path.join(INPUT_DIR, "customers_vanglai.csv")
    path_diamond = os.path.join(INPUT_DIR, "customers_diamond.csv")
    path_standard = os.path.join(INPUT_DIR, "customers_standard.csv")

    # Cú pháp [[id_col]] giúp chỉ chọn cột đó để lưu
    df_vanglai[[id_col]].to_csv(path_vanglai, index=False)
    df_diamond[[id_col]].to_csv(path_diamond, index=False)
    df_standard[[id_col]].to_csv(path_standard, index=False)

    # Lưu model K-means
    joblib.dump(kmeans, os.path.join(INPUT_DIR, "kmeans_rfm_model.joblib"))

    print("\nExport Summary (ID columns only):")
    print(f"   1. Vãng lai (ID=0) : {len(df_vanglai)} IDs -> {path_vanglai}")
    print(f"   2. Diamond (VIP)   : {len(df_diamond)} IDs -> {path_diamond}")
    print(f"   3. Standard        : {len(df_standard)} IDs -> {path_standard}")

    # Also save full labeled data for supervised learning
    df_clean.to_csv(os.path.join(INPUT_DIR, "data_labeled.csv"), index=False)
    print(f"   4. Full labeled data: {len(df_clean)} rows -> {os.path.join(INPUT_DIR, 'data_labeled.csv')}")


def perform_clustering(k=None):
    """Wrapper function for clustering with custom k"""
    clustering_and_split_id_only(k)


def main():
    """Main function for clustering"""
    parser = argparse.ArgumentParser(description='Customer Clustering Pipeline')
    parser.add_argument('--k', type=int, default=BEST_K, help='Number of clusters')
    args = parser.parse_args()

    clustering_and_split_id_only(args.k)


if __name__ == "__main__":
    main()