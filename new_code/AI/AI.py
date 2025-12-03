"""
RFM scoring and customer segmentation
K-means clustering (RFM-score)
Create Expert Loyal label
Train Decision Tree & Random Forest
Author: Trọng Nguyễn Lê
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ========================== CONFIG ==========================
INPUT_FILE = "-ATH-main/src_code/new_dataset/final_merged_data.csv"  # cập nhật path tuyệt đối trong workspace
OUTPUT_DIR = "outputs2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_RANGE = range(2, 7)
# Optionally limit number of rows loaded for faster experimentation; set to an int or leave None
ROW_LIMIT = 500000  # e.g., 50000
# ============================================================


# ============================================================
# 1) Load dữ liệu
# ============================================================
def load_data(rows: int | None = None):
    """
    Đọc final_merged_data.csv:
    transaction_id,date,customer_id,age,gender,country,category,quantity,unit_price,total_amount
    Chuẩn hoá về các cột dùng cho RFM: Customer_ID, Date, Quantity, Total_Amount, Age, Gender.
    Lọc giá trị lỗi (quantity <=0 hoặc total_amount <0).
    """
    file_path = INPUT_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    df = pd.read_csv(file_path)

    # Đổi tên cột
    rename_map = {
        "customer_id": "Customer_ID",
        "date": "Date",
        "quantity": "Quantity",
        "total_amount": "Total_Amount",
        "unit_price": "Unit_Price",
        "category": "Category",
        "age": "Age",
        "gender": "Gender",
        "country": "Country",
        "transaction_id": "InvoiceNo",
        "invoice_no": "InvoiceNo"
    }
    for c_old, c_new in rename_map.items():
        if c_old in df.columns:
            df.rename(columns={c_old: c_new}, inplace=True)

    # Kiểu dữ liệu
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for num_col in ["Quantity", "Total_Amount", "Unit_Price", "Age"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Chuẩn hoá giới tính về mã 0/1 nếu cần (Male=1, Female=0)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype("Int64")

    # Lọc giá trị lỗi
    # before = len(df)
    # df = df[df["Date"].notna()]
    # df = df[df["Quantity"] > 0]
    # df = df[df["Total_Amount"] >= 0]
    # after = len(df)

    # Giới hạn số dòng nếu rows được truyền
    if rows and rows > 0:
        df = df.head(rows)

    # print(f"✅ Đọc {len(df):,} dòng từ {file_path} (lọc bỏ {before - after} dòng lỗi)")
    # missing_core = df[["Customer_ID", "Date", "Quantity", "Total_Amount"]].isna().sum().to_dict()
    # print("🔎 Thiếu dữ liệu lõi:", missing_core)

    return df


# ============================================================
# Utility: take only N rows from a dataframe (head/sample/tail)
# ============================================================
def take_rows(df: pd.DataFrame, n: int, method: str = "head", random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a dataframe limited to n rows.

    method:
      - 'head': first n rows
      - 'tail': last n rows
      - 'sample': random sample of n rows (without replacement)
    """
    if n is None or n <= 0 or n >= len(df):
        return df
    method = (method or "head").lower()
    if method == "tail":
        out = df.tail(n)
    elif method == "sample":
        out = df.sample(n=n, random_state=random_state)
    else:
        out = df.head(n)
    print(f"Using {method} to take {len(out)} rows (from {len(df)})")
    return out


# ============================================================
# 2) Compute RFM
# ============================================================
def compute_rfm(df):
    ref = df["Date"].max() + timedelta(days=1)

    # Determine invoice/transaction column for frequency
    invoice_col = None
    for cand in ["InvoiceNo", "transaction_id", "Transaction_ID"]:
        if cand in df.columns:
            invoice_col = cand
            break

    # Build aggregation dict dynamically
    agg_dict = {"Date": lambda x: (ref - x.max()).days, "Total_Amount": "sum"}
    if invoice_col:
        agg_dict[invoice_col] = "nunique"  # Frequency based on unique invoices/transactions
    else:
        # Fallback: count rows per customer
        df["_row_counter"] = 1
        agg_dict["_row_counter"] = "sum"

    rfm = df.groupby("Customer_ID").agg(agg_dict).reset_index()

    # Rename columns to standard names
    rename_map = {"Date": "Recency", "Total_Amount": "Monetary"}
    if invoice_col:
        rename_map[invoice_col] = "Frequency"
    else:
        rename_map["_row_counter"] = "Frequency"
    rfm = rfm.rename(columns=rename_map)

    # attach age/gender
    demo = df.sort_values("Date").groupby("Customer_ID").last().reset_index()
    # Only merge columns that actually exist in the demo table (Age/Gender may be missing)
    merge_cols = ["Customer_ID"]
    missing = []
    for c in ("Age", "Gender"):
        if c in demo.columns:
            merge_cols.append(c)
        else:
            missing.append(c)

    if missing:
        print(f"Warning: missing columns in input and will be NaN in RFM: {missing}")

    rfm = rfm.merge(demo[merge_cols], on="Customer_ID", how="left")

    return rfm


# ============================================================
# 3) Normalize RFM thành điểm 1–5 bằng phân vị (quantile-based)
# ============================================================
def rfm_quantile_scoring(rfm):
    df = rfm.copy()

    # Recency → nhỏ = tốt → đảo chiều
    try:
        # try direct qcut (works when values have enough unique values)
        df["R_score"] = pd.qcut(df["Recency"], q=5, labels=[5,4,3,2,1])
    except Exception:
        # fallback: bin by rank to avoid issues with duplicate values
        df["R_score"] = pd.qcut(df["Recency"].rank(method="first"), q=5, labels=[5,4,3,2,1])

    # Frequency → lớn = tốt
    try:
        df["F_score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])
    except Exception:
        df["F_score"] = pd.qcut(df["Frequency"].rank(method="first" if "Frequency" in df else "first"), q=5, labels=[1,2,3,4,5])

    # Monetary → lớn = tốt
    try:
        df["M_score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1,2,3,4,5])
    except Exception:
        df["M_score"] = pd.qcut(df["Monetary"].rank(method="first" if "Monetary" in df else "first"), q=5, labels=[1,2,3,4,5])

    # Convert categorical labels to numeric and handle NaNs (e.g., customers with no transactions)
    for col, default in (("R_score", 1), ("F_score", 1), ("M_score", 1)):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(int)

    return df


# ============================================================
# 1) Bảng điểm chuyên gia cho từng RFM Segment
# ============================================================
SEGMENT_SCORE = {
    "Champions": 5.0,
    "Loyal Customers": 4.5,
    "Potential Loyalist": 3.5,
    "Recent Customers": 3.0,
    "Promising": 2.8,
    "Customers Needing Attention": 2.5,
    "About to Sleep": 2.0,
    "At Risk": 1.5,
    "Can't Lose Them": 3.8,
    "Hibernating": 1.2,
    "Lost": 0.5
}


def segment_score(seg):
    return SEGMENT_SCORE.get(seg, 0.5)



# ============================================================
# 2) RFM Segment Rule (từ bạn)
# ============================================================
def rfm_segment(r, f, m):
    # ===== 3 nhóm LOYAL =========
    if r >= 4 and max(f, m) >= 4:
        return "Champions"
    if r >= 2 and max(f, m) >= 3:
        return "Loyal Customers"
    if r >= 3 and 1 <= max(f, m) <= 3:
        return "Potential Loyalist"

    # ===== Các nhóm còn lại =====
    if r >= 4 and max(f, m) <= 1:
        return "Recent Customers"
    if 3 <= r <= 4 and max(f, m) <= 1:
        return "Promising"
    if 2 <= r <= 3 and 2 <= max(f, m) <= 3:
        return "Customers Needing Attention"
    if 2 <= r <= 3 and max(f, m) <= 2:
        return "About to Sleep"
    if r <= 2 and 2 <= max(f, m) <= 5:
        return "At Risk"
    if r <= 1 and 4 <= max(f, m) <= 5:
        return "Can't Lose Them"
    if 1 <= r <= 2 and 1 <= max(f, m) <= 2:
        return "Hibernating"

    return "Lost"



# ============================================================
# 3) Weight Age / Gender
# ============================================================
def gender_weight(g):
    # female có tỉ lệ mua hàng lại cao hơn
    return 0.4 if g == 0 else 0.0


def age_weight(age):
    if age is None or pd.isna(age):
        return 0.0

    age = int(age)

    if 18 <= age <= 25:
        return 0.0
    if 26 <= age <= 35:
        return 0.15
    if 36 <= age <= 45:
        return 0.35     # nhóm chi tiêu tốt nhất
    if age > 45:
        return 0.2

    return 0.0



# ============================================================
# 4) Hàm tạo nhãn Expert Loyal (phiên bản tốt nhất)
# ============================================================
def create_expert_label(rfm):
    """
    Trả về Segment_Name, Segment_Score, Expert_Loyal_Score và nhãn Expert_Loyal (0/1)
    """
    df = rfm.copy()

    required = {"R_score", "F_score", "M_score", "Age", "Gender"}
    if not required.issubset(df.columns):
        raise ValueError("Thiếu cột R_score, F_score, M_score, Age hoặc Gender")

    # ====================================================
    # A) Gán segment theo RFM
    # ====================================================
    df["Segment_Name"] = df.apply(
        lambda x: rfm_segment(x.R_score, x.F_score, x.M_score),
        axis=1
    )

    # B) Gán Segment Score (0–5 điểm)
    df["Segment_Score"] = df["Segment_Name"].apply(segment_score)

    # ====================================================
    # C) Weight từ Age/Gender
    # ====================================================
    df["age_w"] = df["Age"].apply(age_weight)
    df["gender_w"] = df["Gender"].apply(gender_weight)

    # ====================================================
    # D) Công thức Expert Loyalty Score
    # ====================================================
    df["Expert_Loyal_Score"] = (
        df["Segment_Score"]
        + df["age_w"]
        + df["gender_w"]
    )

    # Chuẩn hoá nhãn loyal
    # tối ưu nhất theo nhiều dự án thực tế:
    df["Expert_Loyal"] = df["Expert_Loyal_Score"].apply(lambda x: 1 if x >= 4.0 else 0)

    return df


# ============================================================
# 5) K-means để phân nhóm khách hàng dựa trên RFM-score
# ============================================================
def run_kmeans(df):

    X = df[["R_score", "F_score", "M_score"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = 4   # mặc định
    best_sil = -1

    # tìm K tối ưu
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)

        print(f"k={k} → silhouette={sil:.3f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"✔ BEST K = {best_k}, silhouette={best_sil:.3f}")

    # chạy lại KMeans tốt nhất
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE)
    df["cluster"] = km_final.fit_predict(Xs)

    return df, best_k, best_sil


# ============================================================
# 6) Train Decision Tree & Random Forest
# ============================================================
def train_models(df, use_scores=True):
    """
    Huấn luyện DecisionTree & RandomForest.
    use_scores=True: dùng R_score/F_score/M_score.
    use_scores=False: dùng raw Recency/Frequency/Monetary.
    """
    feature_cols = ["Recency", "Frequency", "Monetary", "Age", "Gender"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột feature: {missing}")

    X = df[feature_cols].fillna(0)
    y = df["Expert_Loyal"]

    X_temp, X_eval, y_temp, y_eval = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=RANDOM_STATE
    )

    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    acc_dt = (y_pred_dt == y_test).mean()
    acc_rf = (y_pred_rf == y_test).mean()

    print(f"\n=== Decision Tree ({'RAW' if not use_scores else 'SCORES'}) ===")
    print("Train acc:", dt.score(X_train, y_train))
    print("Test  acc:", acc_dt)

    print(f"\n=== Random Forest ({'RAW' if not use_scores else 'SCORES'}) ===")
    print("Train acc:", rf.score(X_train, y_train))
    print("Test  acc:", acc_rf)

    joblib.dump(dt, os.path.join(OUTPUT_DIR, f"dt_{'raw' if not use_scores else 'scores'}.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, f"rf_{'raw' if not use_scores else 'scores'}.joblib"))

    plot_prediction_comparison(y_test, y_pred_dt, y_pred_rf)

    return dt, rf


def plot_prediction_comparison(y_true, y_dt, y_rf):
    """Biểu đồ so sánh nhãn thật vs dự đoán (DecisionTree & RandomForest)."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skip prediction comparison plots (matplotlib unavailable)")
        return
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm_dt = confusion_matrix(y_true, y_dt)
    cm_rf = confusion_matrix(y_true, y_rf)

    # Confusion matrices
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion - DecisionTree")
    plt.xlabel("Predicted"); plt.ylabel("Actual")

    plt.subplot(1,2,2)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion - RandomForest")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_confusion_matrices.png"))
    plt.close()

    # Bar chart: counts
    dt_counts = pd.Series(y_dt).value_counts().rename("DecisionTree")
    rf_counts = pd.Series(y_rf).value_counts().rename("RandomForest")
    true_counts = pd.Series(y_true).value_counts().rename("Actual")
    comp = pd.concat([true_counts, dt_counts, rf_counts], axis=1).fillna(0).astype(int)

    comp.to_csv(os.path.join(OUTPUT_DIR, "prediction_label_counts.csv"))

    comp.plot(kind="bar", figsize=(6,4), color=["#1f77b4","#ff7f0e","#2ca02c"])
    plt.title("So sánh số lượng nhãn: Actual vs DT vs RF")
    plt.xlabel("Label (0=Non-Loyal, 1=Loyal)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_label_counts.png"))
    plt.close()

    # Overlay: Loyalty probability (RF)
    try:
        probs = None
        # Nếu rf còn trong scope (gọi hàm bên trong train_models) có thể bỏ qua
    except Exception:
        pass


# (main is defined later after helper functions to avoid NameError on first run)

def prepare_learning_features(rfm: pd.DataFrame, use_scores: bool = False):
    """
    Trả về X,y cho huấn luyện:
    - use_scores = False: dùng raw Recency, Frequency, Monetary, Age, Gender (tránh học trực tiếp quy tắc đã mã hoá).
    - use_scores = True : dùng R_score, F_score, M_score, Age, Gender (pipeline cũ).
    """
    
        # Frequency / Monetary đã ở rfm; Recency cũng có
    base_cols = ["Recency", "Frequency", "Monetary"]
    extra = [c for c in ["Age", "Gender"] if c in rfm.columns]
    feat_cols = base_cols + extra

    missing = [c for c in feat_cols if c not in rfm.columns]
    if missing:
        raise ValueError(f"Thiếu cột cho features: {missing}")

    X = rfm[feat_cols].fillna(0)
    y = rfm["Expert_Loyal"]
    return X, y, feat_cols


def extract_tree_rules(dt: DecisionTreeClassifier, feature_names, max_depth=None):
    """
    Trích xuất rule từ sklearn DecisionTree thành dạng IF … THEN.
    """
    tree_ = dt.tree_
    feat_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, depth, rule):
        if max_depth is not None and depth > max_depth:
            return
        if tree_.feature[node] != -2:
            name = feat_name[node]
            threshold = tree_.threshold[node]
            # Nhánh trái
            recurse(tree_.children_left[node], depth + 1,
                    rule + [f"{name} <= {threshold:.3f}"])
            # Nhánh phải
            recurse(tree_.children_right[node], depth + 1,
                    rule + [f"{name} > {threshold:.3f}"])
        else:
            value = tree_.value[node][0]
            pred = int(np.argmax(value))
            samples = int(np.sum(value))
            paths.append({"rule": " AND ".join(rule),
                          "predict": pred,
                          "samples": samples,
                          "distribution": value.tolist()})

    recurse(0, 0, [])
    # Sắp xếp theo số mẫu giảm dần
    paths = sorted(paths, key=lambda x: -x["samples"])
    lines = []
    for p in paths:
        lines.append(f"IF {p['rule']} THEN Predict={p['predict']} "
                     f"(samples={p['samples']}, dist={p['distribution']})")
    return "\n".join(lines)


def train_derivation_models(rfm: pd.DataFrame, use_scores=False):
    """
    Huấn luyện mô hình trên RAW hoặc SCORE features để tự suy ra 'công thức',
    áp regularization giảm overfit + cross validation cơ bản.
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix

    X, y, feat_cols = prepare_learning_features(rfm, use_scores=use_scores)

    # Split theo thời gian tránh leakage (sắp xếp theo Recency giảm dần giả định)
    temp = rfm.copy()
    temp["_order"] = temp["Recency"].rank(method="first", ascending=False)
    order_index = temp.sort_values("_order").index
    X = X.loc[order_index]
    y = y.loc[order_index]

    n = len(X)
    train_end = int(n * 0.7)
    test_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_hold, y_hold = X.iloc[test_end:], y.iloc[test_end:]

    # Regularized tree & forest
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=25,
        random_state=RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Hold-out
    y_hold_dt = dt.predict(X_hold)
    y_hold_rf = rf.predict(X_hold)

    # Reports
    report_dt = classification_report(y_test, y_pred_dt, output_dict=True)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    cm_dt = confusion_matrix(y_test, y_pred_dt)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    # Trích xuất rule
    rules_txt = extract_tree_rules(dt, feat_cols, max_depth=4)

    # Lưu
    with open(os.path.join(OUTPUT_DIR, "model_derivation_rules.txt"), "w", encoding="utf-8") as f:
        f.write("=== Decision Tree Derived Rules (RAW features) ===\n")
        f.write(rules_txt + "\n")
        f.write("\n=== Feature Importances (RandomForest) ===\n")
        for name, imp in sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1]):
            f.write(f"{name}: {imp:.4f}\n")

    pd.DataFrame(report_dt).to_csv(os.path.join(OUTPUT_DIR, "dt_raw_report.csv"))
    pd.DataFrame(report_rf).to_csv(os.path.join(OUTPUT_DIR, "rf_raw_report.csv"))

    return {
        "dt": dt,
        "rf": rf,
        "feat_cols": feat_cols,
        "report_dt": report_dt,
        "report_rf": report_rf,
        "cm_dt": cm_dt.tolist(),
        "cm_rf": cm_rf.tolist(),
        "y_test": y_test.values,
        "y_pred_dt": y_pred_dt,
        "y_pred_rf": y_pred_rf,
        "y_hold_dt": y_hold_dt,
        "y_hold_rf": y_hold_rf
    }


def compare_expert_vs_model(rfm: pd.DataFrame, model_info: dict):
    """
    So sánh nhãn Expert_Loyal và nhãn mô hình (RandomForest trên RAW).
    Tính tỷ lệ khác biệt & lưu phân phối.
    """
    # Dùng toàn bộ rfm để predict (RF)
    rf = model_info["rf"]
    X_all, _, feat_cols = prepare_learning_features(rfm, use_scores=False)
    model_pred = rf.predict(X_all)
    expert = rfm["Expert_Loyal"].values

    diff = (model_pred != expert).mean()
    agree = 1 - diff

    # Phân phối
    dist_df = pd.DataFrame({
        "Expert_Loyal": expert,
        "Model_Pred": model_pred
    })
    cross = dist_df.value_counts().reset_index(name="count")
    cross.to_csv(os.path.join(OUTPUT_DIR, "model_pred_distribution.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "expert_vs_model_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Agreement Rate: {agree:.4f}\n")
        f.write(f"Disagreement Rate: {diff:.4f}\n")
        f.write("Joint distribution (Expert_Loyal, Model_Pred):\n")
        for _, row in cross.iterrows():
            f.write(f"Expert={row['Expert_Loyal']}, Model={row['Model_Pred']}, Count={row['count']}\n")

    print(f"🔍 Agreement Expert vs Model: {agree:.3f} | Disagreement: {diff:.3f}")
    return cross


def summarize_segments(rfm: pd.DataFrame):
    """Thống kê segment sau khi tạo Expert label.
    Output:
      - segment_summary.csv (có dòng TOTAL cuối cùng)
      - segment_loyal_overview.txt (champions count, total loyal,...)
      - (tuỳ chọn) biểu đồ tỷ lệ loyal và phân bố loyal vs non-loyal.
    """
    if "Segment_Name" not in rfm.columns or "Expert_Loyal" not in rfm.columns:
        print("Segment_Name/Expert_Loyal chưa có, bỏ qua thống kê.")
        return None

    grp = rfm.groupby("Segment_Name").agg(
        total_customers=("Customer_ID", "nunique"),
        loyal=("Expert_Loyal", "sum")
    ).reset_index()
    grp["non_loyal"] = grp["total_customers"] - grp["loyal"]
    grp["loyal_ratio"] = (grp["loyal"] / grp["total_customers"]).round(4)

    # Tổng hợp global
    champion_count = int(grp.loc[grp["Segment_Name"] == "Champions", "total_customers"].sum())
    loyal_total = int(grp["loyal"].sum())
    total_customers = int(grp["total_customers"].sum())
    global_ratio = round(loyal_total / total_customers, 4) if total_customers else 0.0

    # Sắp xếp theo loyal_ratio giảm dần
    grp_sorted = grp.sort_values("loyal_ratio", ascending=False).reset_index(drop=True)
    total_row = pd.DataFrame([{
        "Segment_Name": "TOTAL",
        "total_customers": total_customers,
        "loyal": loyal_total,
        "non_loyal": total_customers - loyal_total,
        "loyal_ratio": global_ratio
    }])
    grp_out = pd.concat([grp_sorted, total_row], ignore_index=True)

    # Xuất CSV
    csv_path = os.path.join(OUTPUT_DIR, "segment_summary.csv")
    grp_out.to_csv(csv_path, index=False)
    print("\n📊 Segment Summary:")
    print(grp_out.to_string(index=False))
    print(f"💾 Lưu CSV: {csv_path}")
    print(f"👑 Champions: {champion_count} | Tổng loyal: {loyal_total} / {total_customers} (ratio={global_ratio:.4f})")

    # Overview text file
    overview_path = os.path.join(OUTPUT_DIR, "segment_loyal_overview.txt")
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("=== SEGMENT LOYAL OVERVIEW ===\n")
        f.write(f"Total customers: {total_customers}\n")
        f.write(f"Champions count: {champion_count}\n")
        f.write(f"Total loyal: {loyal_total}\n")
        f.write(f"Global loyal ratio: {global_ratio:.4f}\n\n")
        f.write("Per segment detail (sorted by loyal_ratio desc):\n")
        for _, row in grp_sorted.iterrows():
            f.write(f"{row['Segment_Name']}: customers={row['total_customers']}, loyal={row['loyal']}, non_loyal={row['non_loyal']}, loyal_ratio={row['loyal_ratio']:.4f}\n")
    print(f"📝 Overview text saved: {overview_path}")

    # Plots
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(10,5))
            plt.bar(grp_sorted["Segment_Name"], grp_sorted["loyal_ratio"], color="#2a9d8f")
            plt.xticks(rotation=50, ha='right')
            plt.ylabel("Tỷ lệ Loyal")
            plt.title("Tỷ lệ Loyal theo Segment (Expert)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "segment_loyal_ratio.png"))
            plt.close()

            plt.figure(figsize=(10,5))
            width = 0.35
            x = np.arange(len(grp_sorted))
            plt.bar(x - width/2, grp_sorted["loyal"], width, label="Loyal", color="#1f77b4")
            plt.bar(x + width/2, grp_sorted["non_loyal"], width, label="Non-Loyal", color="#ff7f0e")
            plt.xticks(x, grp_sorted["Segment_Name"], rotation=50, ha='right')
            plt.ylabel("Số lượng")
            plt.title("Phân bố Loyal vs Non-Loyal theo Segment")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "segment_loyal_nonloyal_counts.png"))
            plt.close()
        except Exception as e:
            print(f"(Không vẽ được biểu đồ segment: {e})")
    else:
        print("(Thiếu matplotlib: bỏ qua vẽ biểu đồ segment)")

    return grp_out

# --- Cập nhật main: thêm bước derivation & comparison ---
def main():
    df = load_data(ROW_LIMIT)
    rfm = compute_rfm(df)
    rfm = rfm_quantile_scoring(rfm)
    rfm = create_expert_label(rfm)
    # Thống kê segment ngay sau gán nhãn Expert
    summarize_segments(rfm)
    rfm_clustered, best_k, best_sil = run_kmeans(rfm)

    # Train với RAW Recency/Frequency/Monetary (không dùng R_score/F_score/M_score)
    dt_raw, rf_raw = train_models(rfm_clustered, use_scores=False)

    # Mô hình suy công thức trên RAW (giữ nguyên nếu cần phân tích thêm)
    print("\n================= DERIVATION (RAW FEATURES) =================")
    deriv_info = train_derivation_models(rfm_clustered, use_scores=False)
    compare_expert_vs_model(rfm_clustered, deriv_info)

    rfm_clustered.to_csv(os.path.join(OUTPUT_DIR, "rfm_scores.csv"), index=False)
    print("\n🎯 DONE. Kiểm tra outputs2/ để xem:")
    print("- model_derivation_rules.txt (các rule Decision Tree tự suy ra)")
    print("- expert_vs_model_metrics.txt (tỷ lệ khác biệt với nhãn Expert)")
    print("- model_pred_distribution.csv (bảng kết hợp Expert vs Model)")
    print(f"KMeans best_k={best_k}, silhouette={best_sil:.3f}")

if __name__ == "__main__":
    main()