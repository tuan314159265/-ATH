import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
INPUT_DIR = "pipeline_data"
OUTPUT_DIR = "visualize_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình giao diện
sns.set(style="whitegrid")
my_palette = sns.color_palette("pastel")

def load_and_merge_data():
    print("🔌 Connecting to Database & Merging Data...")
    
    # 1. Load dữ liệu phân cụm (đã gán nhãn)
    try:
        df_labeled = pd.read_csv(os.path.join(INPUT_DIR, "data_labeled.csv"))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file 'pipeline_data/data_labeled.csv'. Hãy chạy bước phân cụm trước.")
        return None

    # 2. Load dữ liệu giao dịch chi tiết từ DB
    engine = create_engine(DB_URI)
    # Lưu ý: Dùng "AS date" để đảm bảo tên cột rõ ràng
    query = """
    SELECT 
        c.customer_id, 
        f.transaction_id, 
        cat.category,
        d.date AS transaction_date
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_category cat ON f.category_id = cat.category_id
    JOIN dim_date d ON f.date_id = d.date_id
    WHERE f.total_amount > 0
    """
    print("   -> Executing SQL...")
    df_trans = pd.read_sql(query, engine)
    
    # --- QUAN TRỌNG: Chuẩn hóa tên cột ---
    df_trans.columns = df_trans.columns.str.lower() # Đưa hết về chữ thường
    
    # Chuyển đổi ngày tháng
    if 'transaction_date' in df_trans.columns:
        df_trans['date'] = pd.to_datetime(df_trans['transaction_date'])
    
    # 3. Merge: Gán nhãn Segment cho từng giao dịch
    # Inner join để chỉ lấy các giao dịch của những khách hàng đã được phân cụm
    df_full = df_trans.merge(df_labeled[['customer_id', 'Segment']], on='customer_id', how='inner')
    
    print(f"   -> Data Ready: {len(df_full)} transactions merged.")
    return df_full, df_labeled

# --- BIỂU ĐỒ 1: PHÂN BỐ KHÁCH HÀNG (Fig 10) ---
def plot_fig10_distribution(df_labeled):
    print("📊 Drawing Fig 10: Segment Distribution...")
    counts = df_labeled['Segment'].value_counts()
    
    plt.figure(figsize=(10, 8))
    explode = [0.05 if c == counts.min() else 0 for c in counts]
    
    plt.pie(
        counts, labels=counts.index, autopct='%1.1f%%', 
        startangle=140, colors=my_palette, explode=explode, shadow=True,
        textprops={'fontsize': 12}
    )
    plt.title('Customer Segmentation Distribution', fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_cluster_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- BIỂU ĐỒ 2: SỐ GIAO DỊCH THEO DANH MỤC (Fig 11) ---
def plot_fig11_category_bar(df_full):
    print("📊 Drawing Fig 11: Category Transactions...")
    
    # Lấy Top 3 Category lớn nhất
    top_cats = df_full['category'].value_counts().head(3).index
    df_filter = df_full[df_full['category'].isin(top_cats)]
    
    # Groupby
    data = df_filter.groupby(['category', 'Segment'])['transaction_id'].count().reset_index()
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=data, x='category', y='transaction_id', hue='Segment',
        palette="pastel", edgecolor="black", alpha=0.9
    )
    
    # Gán nhãn số
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3)
        
    plt.title('Transactions by Category & Segment', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Transactions')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_category_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- BIỂU ĐỒ 3: HEATMAP SỞ THÍCH (Fig 12) ---
def plot_fig12_heatmap(df_full):
    print("📊 Drawing Fig 12: Preference Heatmap...")
    
    pivot = df_full.pivot_table(
        index='category', columns='Segment', values='transaction_id', 
        aggfunc='count', fill_value=0
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='d', cmap="YlGnBu", linewidths=.5)
    plt.title('Transaction Heatmap: Category vs Segment', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig12_heatmap_category.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- BIỂU ĐỒ 4: BOXPLOT VÒNG ĐỜI (Fig 14) ---
def plot_fig14_lifespan(df_full):
    print("📦 Drawing Fig 14: Customer Lifespan...")
    
    # Tính Lifespan: Max Date - Min Date
    lifespan = df_full.groupby(['customer_id', 'Segment'])['date'].agg(['min', 'max']).reset_index()
    lifespan['days'] = (lifespan['max'] - lifespan['min']).dt.days
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=lifespan, x='Segment', y='days', palette="Set2", showfliers=False)
    
    plt.title('Customer Lifespan Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('Lifespan (Days)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "fig14_boxplot_lifespan.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load dữ liệu
    data = load_and_merge_data()
    
    if data:
        df_full, df_labeled = data
        
        # 2. Vẽ lần lượt 4 biểu đồ
        plot_fig10_distribution(df_labeled)
        plot_fig11_category_bar(df_full)
        plot_fig12_heatmap(df_full)
        plot_fig14_lifespan(df_full)
        
        print(f"\n✅ HOÀN TẤT! 4 biểu đồ đã được lưu tại: {OUTPUT_DIR}/")