import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
INPUT_DIR = "pipeline_data"
OUTPUT_DIR = "visualize_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình giao diện
plt.style.use('seaborn-v0_8-whitegrid')

def plot_elbow_method():
    print("📈 Drawing Elbow Method (using rfm_scaled.csv)...")
    
    # 1. Load dữ liệu đã Scale từ File 1
    # Đây là dữ liệu CHÍNH XÁC mà mô hình đã học
    try:
        df_scaled = pd.read_csv(os.path.join(INPUT_DIR, "rfm_scaled.csv"))
    except FileNotFoundError:
        print("❌ Thiếu file 'rfm_scaled.csv'. Hãy chạy File 1 trước.")
        return

    # 2. Chạy vòng lặp K
    sse = []
    k_range = range(1, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
        
    # 3. Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2980b9')
    
    plt.title('The Elbow Method (Optimal K)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('SSE (Inertia)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mũi tên chỉ vào K=4 (Bạn có thể sửa thành K khác nếu muốn)
    optimal_k = 2
    plt.annotate(f'Optimal K = {optimal_k}', 
                 xy=(optimal_k, sse[optimal_k-1]), 
                 xytext=(optimal_k + 2, sse[optimal_k-1] + 1000), # Chỉnh vị trí chữ
                 arrowprops=dict(facecolor='#c0392b', shrink=0.05),
                 fontsize=12, color='#c0392b', ha='center')
    
    # Lưu ảnh
    save_path = os.path.join(OUTPUT_DIR, "elbow_method_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Elbow Chart saved to: {save_path}")
    plt.close()

def plot_3d_clusters():
    print("🧊 Drawing 3D Scatter Plot (using data_labeled.csv)...")
    
    # 1. Load dữ liệu đã gán nhãn từ File 2
    # Dùng dữ liệu gốc (chưa scale) để hiển thị trục số thực (VD: Tiền là $)
    try:
        df = pd.read_csv(os.path.join(INPUT_DIR, "data_labeled.csv"))
    except FileNotFoundError:
        print("❌ Thiếu file 'data_labeled.csv'. Hãy chạy File 2 trước.")
        return
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Màu sắc cho từng nhóm (Diamond -> Bronze)
    # Đỏ, Xanh lá, Xanh dương, Vàng
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f', '#9b59b6']
    
    # Lấy danh sách các cụm đã sắp xếp (để Diamond luôn hiện trước)
    # Sắp xếp theo Segment name để legend đẹp hơn
    segments = sorted(df['Segment'].unique())
    
    for i, segment in enumerate(segments):
        subset = df[df['Segment'] == segment]
        
        ax.scatter(
            subset['Recency'], 
            subset['Frequency'], 
            subset['Monetary'],
            c=colors[i % len(colors)],
            label=segment,
            s=50,    # Kích thước điểm
            alpha=0.6, # Độ trong suốt
            edgecolors='w', linewidth=0.5
        )

    # Trang trí trục
    ax.set_xlabel('Recency (Days)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Frequency (Orders)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Monetary ($)', fontsize=11, fontweight='bold', labelpad=10)
    
    ax.set_title('Customer Segments in 3D RFM Space', fontsize=16, y=1.02)
    
    # Góc nhìn đẹp nhất
    ax.view_init(elev=30, azim=135)
    
    plt.legend(title="Segments", loc="upper left", bbox_to_anchor=(0, 0.9))
    plt.tight_layout()
    
    # Lưu ảnh
    save_path = os.path.join(OUTPUT_DIR, "rfm_3d_clusters.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D Chart saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_elbow_method()
    plot_3d_clusters()