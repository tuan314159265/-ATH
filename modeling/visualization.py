"""
Visualization Module

This module contains all visualization functions for the customer analytics pipeline,
including clustering plots, model comparisons, and data exploration charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
import seaborn as sns
import os
from sqlalchemy import create_engine
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import warnings
import argparse

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
INPUT_DIR = "data"
OUTPUT_DIR = "visualize_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình giao diện
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(style="whitegrid")
my_palette = sns.color_palette("pastel")


def plot_elbow_method():
    """Plot elbow method to find optimal K for K-means"""
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
    """Plot 3D scatter plot of customer segments in RFM space"""
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


def get_clean_data():
    """Get clean RFM data from database"""
    print("🔌 Connecting to Database...")
    engine = create_engine(DB_URI)
    query = """
    SELECT c.customer_id, d.date, f.transaction_id, f.total_amount
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_date d ON f.date_id = d.date_id
    WHERE f.total_amount > 0
    """
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])

    # 1. FIX LỖI THỜI GIAN (Date Gap)
    # Dữ liệu UCI cũ (2011) cần được kéo về hiện tại (2024)
    print("⏳ Fixing Date Gap...")
    def shift_date(x):
        if x.year < 2015: return x.replace(year = x.year + 13)
        return x
    df['date'] = df['date'].apply(shift_date)

    # 2. TÍNH RFM
    print("📐 Calculating RFM...")
    ref_date = df['date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (ref_date - x.max()).days,
        'transaction_id': 'nunique',
        'total_amount': 'sum'
    }).rename(columns={'date': 'Recency', 'transaction_id': 'Frequency', 'total_amount': 'Monetary'})

    # 3. KHỬ NHIỄU (IQR) TRƯỚC KHI BOX-COX
    print("🧹 Removing Outliers...")
    for c in ['Recency', 'Frequency', 'Monetary']:
        Q1 = rfm[c].quantile(0.05)
        Q3 = rfm[c].quantile(0.95)
        IQR = Q3 - Q1
        rfm = rfm[(rfm[c] >= Q1 - 1.5*IQR) & (rfm[c] <= Q3 + 1.5*IQR)]

    return rfm


def plot_boxcox_transformation():
    """Plot before/after Box-Cox transformation"""
    print("📊 Drawing Box-Cox Transformation Charts...")

    rfm = get_clean_data()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        # Original distribution
        axes[0, i].hist(rfm[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, i].set_title(f'Original {col}')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')

        # Q-Q plot for normality check
        stats.probplot(rfm[col], dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot: {col}')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "original_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Original distributions saved to: {save_path}")
    plt.close()

    # After Box-Cox
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        # Apply Box-Cox
        transformed, _ = stats.boxcox(rfm[col] + 1)  # +1 to handle zeros

        # Transformed distribution
        axes[0, i].hist(transformed, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, i].set_title(f'Box-Cox {col}')
        axes[0, i].set_xlabel(f'Transformed {col}')
        axes[0, i].set_ylabel('Frequency')

        # Q-Q plot after transformation
        stats.probplot(transformed, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot: Box-Cox {col}')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "boxcox_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Box-Cox distributions saved to: {save_path}")
    plt.close()


def load_and_merge_data():
    """Load and merge labeled data with transaction details"""
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

    # 3. Merge với dữ liệu phân cụm
    df_merged = df_trans.merge(
        df_labeled[['customer_id', 'segment', 'cluster']],
        on='customer_id',
        how='left'
    )

    return df_merged


def plot_segment_analysis():
    """Plot comprehensive segment analysis"""
    print("📊 Drawing Segment Analysis Charts...")

    df = load_and_merge_data()
    if df is None:
        return

    # 1. Segment Distribution
    plt.figure(figsize=(10, 6))
    segment_counts = df['segment'].value_counts()
    segment_counts.plot(kind='bar', color=['#e74c3c', '#2ecc71', '#3498db'])
    plt.title('Customer Segment Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(segment_counts):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, "segment_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Segment distribution saved to: {save_path}")
    plt.close()

    # 2. Category Preferences by Segment
    plt.figure(figsize=(12, 8))
    category_segment = pd.crosstab(df['category'], df['segment'], normalize='index') * 100
    category_segment.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Category Preferences by Customer Segment', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "category_preferences.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Category preferences saved to: {save_path}")
    plt.close()

    # 3. Monthly Trends by Segment
    df['month'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')
    monthly_segment = df.groupby(['month', 'segment']).size().unstack().fillna(0)

    plt.figure(figsize=(15, 8))
    monthly_segment.plot(marker='o', linewidth=2, markersize=6)
    plt.title('Monthly Transaction Trends by Segment', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "monthly_trends.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Monthly trends saved to: {save_path}")
    plt.close()


def plot_confusion(y_true, y_pred, classes, title, out_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importances(importances, features, title, out_path, top_n=10):
    """Plot feature importances"""
    idx = np.argsort(importances)[-top_n:][::-1]
    top_feats = [features[i] for i in idx]
    top_vals = importances[idx]
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_vals, y=top_feats)
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pca_compare(X, y_true, y_dt, y_rf, classes, out_base):
    """Plot PCA comparison of true vs predicted labels"""
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax, lab, col in zip(axes, [y_true, y_dt, y_rf], ['True', 'DT', 'RF']):
        sc = ax.scatter(Z[:,0], Z[:,1], c=lab, cmap='tab10', s=10)
        ax.set_title(col)
    plt.suptitle('PCA projection: True vs DT vs RF')
    plt.tight_layout(rect=[0,0,1,0.95])
    out_path = os.path.join(out_base, 'pca_compare.png')
    plt.savefig(out_path)
    plt.close()


def compare_models(labeled_csv=None, target='cluster', save_dir=None):
    """Compare Decision Tree and Random Forest models"""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data')
    LABELED_CSV = labeled_csv or os.path.join(PIPELINE_DIR, 'data_labeled.csv')
    SCALED_CSV = os.path.join(PIPELINE_DIR, 'rfm_scaled.csv')

    out_dir = save_dir or os.path.join(PROJECT_ROOT, 'outputs', 'model_comparison')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(LABELED_CSV):
        raise FileNotFoundError(f"Labeled CSV not found: {LABELED_CSV}")

    df = pd.read_csv(LABELED_CSV)

    # Prepare features
    drop = {'customer_id', 'Cluster', 'Segment'}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in drop]
    X = df[features].copy()

    if target == 'cluster':
        y = df['Cluster'].values
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df['Segment'].astype(str).values)

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train models
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_test)

    y_dt = dt.predict(X_test)
    y_rf = rf.predict(X_test)

    print('Decision Tree report:')
    print(classification_report(y_test, y_dt, zero_division=0))
    print('Random Forest report:')
    print(classification_report(y_test, y_rf, zero_division=0))

    classes = np.unique(y)

    # Confusion matrices
    plot_confusion(y_test, y_dt, classes, 'Decision Tree Confusion', os.path.join(out_dir, 'confusion_dt.png'))
    plot_confusion(y_test, y_rf, classes, 'Random Forest Confusion', os.path.join(out_dir, 'confusion_rf.png'))

    # Feature importances
    plot_feature_importances(rf.feature_importances_, features, 'Random Forest Importances', os.path.join(out_dir, 'feat_imp_rf.png'))
    try:
        plot_feature_importances(dt.feature_importances_, features, 'Decision Tree Importances', os.path.join(out_dir, 'feat_imp_dt.png'))
    except Exception:
        pass

    # PCA comparison
    X_test_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full_dt = np.concatenate([dt.predict(X_train), dt.predict(X_test)])
    y_full_rf = np.concatenate([rf.predict(X_train), rf.predict(X_test)])
    y_full_true = np.concatenate([y_train, y_test])
    plot_pca_compare(X_test_full.values, y_full_true, y_full_dt, y_full_rf, classes, out_dir)

    # Silhouette score
    if os.path.exists(SCALED_CSV):
        X_scaled = pd.read_csv(SCALED_CSV)
        if 'customer_id' in X_scaled.columns and 'customer_id' in df.columns:
            merged = pd.merge(df[['customer_id', 'Cluster']], X_scaled, on='customer_id', how='left')
            labels = merged['Cluster'].values
            X_for_sil = merged.drop(columns=['customer_id', 'Cluster']).values
            try:
                s = silhouette_score(X_for_sil, labels)
                print(f'Silhouette score (clusters): {s:.4f}')
            except Exception as e:
                print('Silhouette score error:', e)
        else:
            try:
                s = silhouette_score(X.values, df['Cluster'].values)
                print(f'Silhouette score (clusters): {s:.4f}')
            except Exception as e:
                print('Silhouette score error:', e)
    else:
        try:
            s = silhouette_score(X.values, df['Cluster'].values)
            print(f'Silhouette score (clusters): {s:.4f}')
        except Exception as e:
            print('Silhouette score error:', e)

    print(f'Saved plots and models to {out_dir}')


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='Customer Analytics Visualization')
    parser.add_argument('--action', choices=['elbow', '3d', 'boxcox', 'segments', 'compare', 'all'],
                       default='all', help='Visualization to generate')
    parser.add_argument('--labeled', help='Path to labeled data CSV')
    parser.add_argument('--save-dir', help='Output directory for model comparison')
    args = parser.parse_args()

    if args.action in ['elbow', 'all']:
        plot_elbow_method()

    if args.action in ['3d', 'all']:
        plot_3d_clusters()

    if args.action in ['boxcox', 'all']:
        plot_boxcox_transformation()

    if args.action in ['segments', 'all']:
        plot_segment_analysis()

    if args.action in ['compare', 'all']:
        compare_models(args.labeled, save_dir=args.save_dir)


if __name__ == "__main__":
    main()