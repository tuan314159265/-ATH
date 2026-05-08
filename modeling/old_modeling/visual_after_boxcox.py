import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
from sqlalchemy import create_engine
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import warnings

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
DB_URI = "postgresql+psycopg2://tuan:123@localhost:5432/ord"
OUTPUT_DIR = "visualize_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid")

def get_clean_data():
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
    ref_date = df['date'].max() + timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (ref_date - x.max()).days,
        'transaction_id': 'nunique',
        'total_amount': 'sum'
    }).rename(columns={'date': 'Recency', 'transaction_id': 'Frequency', 'total_amount': 'Monetary'})
    
    # 3. KHỬ NHIỄU (IQR) TRƯỚC KHI BOX-COX
    # Box-Cox rất nhạy cảm với nhiễu, nên cần lọc trước
    print("🧹 Removing Outliers (IQR)...")
    for col in rfm.columns:
        Q1 = rfm[col].quantile(0.05)
        Q3 = rfm[col].quantile(0.95)
        IQR = Q3 - Q1
        rfm = rfm[(rfm[col] >= Q1 - 1.5 * IQR) & (rfm[col] <= Q3 + 1.5 * IQR)]
        
    return rfm

def apply_boxcox_and_plot(rfm):
    print("📊 Applying Box-Cox & Plotting...")
    
    cols = ['Recency', 'Frequency', 'Monetary']
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # Khởi tạo PowerTransformer (Yeo-Johnson mạnh hơn Box-Cox vì xử lý được số 0)
    pt = PowerTransformer(method='yeo-johnson')
    
    for i, col in enumerate(cols):
        # --- CỘT TRÁI: DỮ LIỆU GỐC ---
        original_data = rfm[[col]]
        skew_orig = original_data[col].skew()
        
        sns.histplot(data=original_data, x=col, kde=True, ax=axes[i, 0], color='#3498db', bins=30)
        axes[i, 0].set_title(f'Original {col}\nSkewness: {skew_orig:.2f}', fontweight='bold')
        
        # --- CỘT PHẢI: SAU KHI BOX-COX ---
        # Biến đổi dữ liệu
        transformed_data = pt.fit_transform(original_data)
        # Chuyển về DataFrame để dễ vẽ
        transformed_df = pd.DataFrame(transformed_data, columns=[col])
        skew_trans = transformed_df[col].skew()
        
        sns.histplot(data=transformed_df, x=col, kde=True, ax=axes[i, 1], color='#e74c3c', bins=30)
        axes[i, 1].set_title(f'After Box-Cox/Yeo-Johnson {col}\nSkewness: {skew_trans:.2f}', fontweight='bold')
        
        print(f"   -> {col}: Skewness improved from {skew_orig:.2f} to {skew_trans:.2f}")

    plt.suptitle('Impact of Box-Cox Transformation on RFM Distribution', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, "rfm_boxcox_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Image saved to: {save_path}")
    # plt.show() # Bật dòng này nếu muốn xem cửa sổ popup

if __name__ == "__main__":
    rfm_df = get_clean_data()
    apply_boxcox_and_plot(rfm_df)