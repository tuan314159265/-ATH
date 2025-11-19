import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
INPUT_DIR = "pipeline_data"


def clustering_and_labeling(k):
    print(f"Running K-Means with K={k}...")
    
    # Load data
    data_pca = np.load(os.path.join(INPUT_DIR, "data_pca.npy"))
    df_clean = pd.read_csv(os.path.join(INPUT_DIR, "data_clean.csv"))
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_pca)
    
    # Gán cluster vào data gốc
    df_clean['Cluster'] = clusters
    
    # --- TỰ ĐỘNG ĐẶT TÊN LABEL ---
    print("🏷️ Assigning Business Labels...")
    summary = df_clean.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Tính điểm để xếp hạng (R thấp tốt, F cao tốt, M cao tốt)
    summary['Score'] = (summary['Recency'].rank(ascending=False) + 
                        summary['Frequency'].rank(ascending=True) + 
                        summary['Monetary'].rank(ascending=True))
    summary = summary.sort_values('Score', ascending=False)
    
    # Map tên (Ví dụ với K=4)
    # Bạn có thể custom lại tên này
    names_map = {
        summary.index[0]: "Champions (VIP)",
        summary.index[1]: "Potential Loyalists",
        summary.index[2]: "At-Risk / Hibernating",
        summary.index[3]: "Lost / Low Value"
    }


    df_clean['Segment'] = df_clean['Cluster'].map(names_map)
    
    print("\n📊 Cluster Summary:")
    print(summary)
    print("\n👉 Segment Mapping:", names_map)
    
    # Lưu kết quả
    df_clean.to_csv(os.path.join(INPUT_DIR, "data_labeled.csv"), index=False)
    joblib.dump(kmeans, os.path.join(INPUT_DIR, "kmeans_model.joblib"))
    print("✅ Labeled data saved!")

if __name__ == "__main__":
    clustering_and_labeling()