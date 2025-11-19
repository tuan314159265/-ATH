import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

# --- CONFIG ---
INPUT_DIR = "pipeline_data"

def train_models():
    print("🤖 Loading Labeled Data...")
    df = pd.read_csv(os.path.join(INPUT_DIR, "data_labeled.csv"))
    
    # Loại bỏ các cột không dùng để train (như ID, Cluster số)
    # Giữ lại RFM, Category %, Age, Gender để train
    X = df.drop(columns=['customer_id', 'Cluster', 'Segment'])
    y = df['Segment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ==========================================
    # PART A: DECISION TREE (MINH HỌA - WHITE BOX)
    # ==========================================
    print("\n🌳 Training Decision Tree (For Explanation)...")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42) # Depth thấp để dễ nhìn
    dt.fit(X_train, y_train)
    
    print("   Decision Tree Accuracy:", dt.score(X_test, y_test))
    
    # Xuất luật ra màn hình (Dùng để báo cáo)
    rules = export_text(dt, feature_names=list(X.columns))
    print("\n--- RULES EXTRACTED FROM DECISION TREE ---")
    print(rules)
    print("------------------------------------------")
    
    # ==========================================
    # PART B: RANDOM FOREST (DỰ ĐOÁN - BLACK BOX)
    # ==========================================
    print("\n🌲 Training Random Forest Pipeline (For Prediction)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Random Forest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance (Yếu tố nào quan trọng nhất?)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n📊 Top 5 Important Features:")
    print(importances.head(5))
    
    # Lưu Model
    joblib.dump(rf, os.path.join(INPUT_DIR, "rf_production_model.joblib"))
    print(f"\n✅ Production Model saved to {INPUT_DIR}/rf_production_model.joblib")

if __name__ == "__main__":
    train_models()