import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- CONFIG ---
INPUT_DIR = "pipeline_data"

def train_models():
    print("🤖 Loading Labeled Data...")
    df = pd.read_csv(os.path.join(INPUT_DIR, "data_labeled.csv"))
    
    # --- CHUẨN BỊ DỮ LIỆU TRAIN ---
    # Loại bỏ các cột định danh, giữ lại các features số để máy học
    # Lưu ý: 'Segment' là Target (Y), 'Cluster' và 'customer_id' bỏ đi
    drop_cols = ['customer_id', 'Cluster', 'Segment']
    
    X = df.drop(columns=drop_cols)
    y = df['Segment'] # Nhãn phức tạp (VD: Diamond | GenZ Women | Fashion)
    
    print(f"   Features used: {list(X.columns)}")
    print(f"   Target classes: {y.nunique()} segments")

    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ==========================================================================
    # PART A: DECISION TREE (MINH HỌA LOGIC PHỨC TẠP)
    # ==========================================================================
    print("\n🌳 Training Decision Tree (To Explain Logic)...")
    
    # Tăng max_depth lên 4 hoặc 5 để cây có thể 'hiểu' được cả RFM lẫn Category
    dt = DecisionTreeClassifier(max_depth=4, random_state=42) 
    dt.fit(X_train, y_train)
    
    print(f"   Decision Tree Accuracy: {dt.score(X_test, y_test):.4f}")
    
    # Xuất luật ra màn hình
    # Luật này sẽ cho bạn thấy máy nó 'nhìn' vào đâu để phân loại
    # VD: Nếu Monetary > 5000 -> Diamond; Nếu Pct_Fashion > 0.5 -> Fashion
    rules = export_text(dt, feature_names=list(X.columns))
    print("\n--- EXPLAINABLE RULES (WHITE BOX) ---")
    print(rules)
    print("-------------------------------------")
    
    # ==========================================================================
    # PART B: RANDOM FOREST (MODEL DỰ ĐOÁN CHÍNH)
    # ==========================================================================
    print("\n🌲 Training Random Forest Pipeline (For Deployment)...")
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Random Forest Accuracy: {acc:.4f}")
    
    # Báo cáo chi tiết
    # Vì tên label dài, ta in classification report để xem độ chính xác từng nhóm
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance (Yếu tố nào định hình nên Persona?)
    # Ví dụ: Để phân biệt nhóm "Tech" vs "Fashion", cột Pct_Technology sẽ quan trọng
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n📊 Top 8 Important Features:")
    print(importances.head(8))
    
    # Lưu Model
    joblib.dump(rf, os.path.join(INPUT_DIR, "rf_production_model.joblib"))
    print(f"\n✅ Production Model saved to {INPUT_DIR}/rf_production_model.joblib")
    
    # --- DEMO DỰ ĐOÁN KHÁCH MỚI ---
    print("\n🔮 Demo Prediction (Test Case):")
    # Lấy thử 1 khách hàng từ tập test để dự đoán
    sample_customer = X_test.iloc[0:1]
    true_label = y_test.iloc[0]
    pred_label = rf.predict(sample_customer)[0]
    
    print(f"   Features: {sample_customer.to_dict(orient='records')[0]}")
    print(f"   True Label:      {true_label}")
    print(f"   Predicted Label: {pred_label}")

if __name__ == "__main__":
    train_models()