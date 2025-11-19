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

# ========================== CONFIG ==========================
INPUT_FILE = "new_dataset/transform_data_filled.csv"
OUTPUT_DIR = "outputs2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
K_RANGE = range(2, 7)
# ============================================================


# ============================================================
# 1) Load dữ liệu
# ============================================================
def load_data():
    print(f"📂 Loading data from: {INPUT_FILE}")

    # Try primary path, then a likely alternate path under src_code
    paths_to_try = [INPUT_FILE, os.path.join("src_code", INPUT_FILE), os.path.join("-ATH-main", "src_code", INPUT_FILE)]
    file_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Generate it by running the filler script or set INPUT_FILE to the correct path.\n"
            "Example generator: -ATH-main/src_code/new_dataset/fill_data.py"
        )

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    else:
        # fallback: try CSV then Excel
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_excel(file_path)

    # --- Normalize column-name lookup (preserve original names) ---
    col_map = {c.lower(): c for c in df.columns}

    def find_col(cands):
        for cand in cands:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    # Candidate names for transaction date and customer date fields
    trans_date_candidates = ["Date", "date", "InvoiceDate", "invoice_date", "Invoice Date", "order_date", "OrderDate"]
    dob_candidates = ["dob", "date_of_birth", "Date of Birth", "BirthDate", "birth_date"]
    reg_candidates = ["registration_date", "registration", "registered", "signup_date"]

    # Parse and normalize any date-like columns found
    parsed_date_cols = []
    for name, cands in (("Date", trans_date_candidates), ("dob", dob_candidates), ("registration_date", reg_candidates)):
        orig = find_col(cands)
        if orig:
            # parse with dayfirst=True (dataset appears UK-style) and coerce errors
            df[orig] = pd.to_datetime(df[orig], errors="coerce", dayfirst=True)
            # strip time component (keep date at midnight)
            df[orig] = df[orig].dt.normalize()
            parsed_date_cols.append(orig)
            # map to canonical names used elsewhere
            if name == "Date":
                df["Date"] = df[orig]
            else:
                df[name] = df[orig]

    # If no explicit transaction Date found, try to auto-detect a date column by sampling
    if "Date" not in df.columns or df["Date"].isna().all():
        for c in df.columns:
            if c.lower() in ("date", "invoice_date", "invoicedate", "order_date", "orderdate"):
                continue  # already tried common names above
            # try lightweight parse on sample
            try:
                sample = pd.to_datetime(df[c].dropna().head(20), errors="coerce", dayfirst=True)
                if sample.notna().sum() >= max(1, int(len(sample) * 0.2)):
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                    df[c] = df[c].dt.normalize()
                    df["Date"] = df[c]
                    parsed_date_cols.append(c)
                    break
            except Exception:
                continue

    # Coerce other expected types (existing logic)
    # --- Normalize customer/invoice/quantity column names to canonical names used later ---
    customer_candidates = ["Customer_ID", "Customer Id", "CustomerID", "Customer Id", "customer_id", "customer id", "Customer", "CustID", "cust_id", "CustomerNo", "Customer No", "CustomerNumber"]
    invoice_candidates = ["InvoiceNo", "Invoice No", "Invoice", "invoice_no", "InvoiceNumber", "Invoice Number", "OrderNo", "Order ID"]
    qty_candidates = ["Quantity", "quantity", "Qty", "QTY"]

    cust_col = find_col(customer_candidates)
    if cust_col and cust_col != "Customer_ID":
        df = df.rename(columns={cust_col: "Customer_ID"})

    inv_col = find_col(invoice_candidates)
    if inv_col and inv_col != "InvoiceNo":
        df = df.rename(columns={inv_col: "InvoiceNo"})

    qty_col = find_col(qty_candidates)
    if qty_col and qty_col != "Quantity":
        df = df.rename(columns={qty_col: "Quantity"})

    # Coerce other expected types (existing logic)
    if "Gender" in df.columns:
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Normalize common monetary column name to Total_Amount if possible
    tot_candidates = ["Total_Amount", "Total Amount", "total_amount", "total amount", "TotalAmount"]
    tot_col = find_col(tot_candidates)
    if tot_col:
        df["Total_Amount"] = pd.to_numeric(df[tot_col], errors="coerce")
    elif "Total_Amount" not in df.columns and "total_amount" in col_map:
        df["Total_Amount"] = pd.to_numeric(df[col_map["total_amount"]], errors="coerce")

    print("Parsed date columns:", parsed_date_cols if parsed_date_cols else "none found")
    if "Date" not in df.columns or df["Date"].isna().all():
        print("Warning: transaction Date not found or all NaT. Ensure input has a valid transaction date column.")

    print(f"✔ Loaded {len(df)} rows from {file_path}")
    return df


# ============================================================
# 2) Compute RFM
# ============================================================
def compute_rfm(df):

    ref = df["Date"].max() + timedelta(days=1)

    rfm = df.groupby("Customer_ID").agg({
        "Date": lambda x: (ref - x.max()).days,     # Recency
        "InvoiceNo": "nunique",                     # Frequency
        "Total_Amount": "sum"                       # Monetary
    }).reset_index()

    rfm = rfm.rename(columns={
        "Date": "Recency",
        "InvoiceNo": "Frequency",
        "Total_Amount": "Monetary"
    })

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
# 4) Tạo nhãn Expert Loyal theo RFM-score
# ============================================================
def rfm_segment(r, f, m):
    # ===== 3 nhóm LOYAL =========
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 4 and f >= 3:
        return "Loyal Customers"
    if r >= 3 and f >= 3:
        return "Potential Loyalist"

    # ===== Các nhóm còn lại =====
    if r >= 4 and f <= 2:
        return "Recent Customers"
    if r == 3 and f <= 2:
        return "Promising"
    if r == 2 and f >= 3:
        return "Customers Needing Attention"
    if r == 2 and f <= 2:
        return "About to Sleep"
    if r == 1 and f >= 4:
        return "At Risk"
    if r == 1 and f == 3:
        return "Can't Lose Them"
    if r == 1 and f == 2:
        return "Hibernating"
    return "Lost"


# ==========================================
# 5. MAP → NHÃN NHỊ PHÂN: LOYAL = 1 / NON-LOYAL = 0
# ==========================================
def map_loyal(seg):
    loyal = {"Champions", "Loyal Customers", "Potential Loyalist"}
    return 1 if seg in loyal else 0


def create_expert_label(rfm):
    """Create `Segment_Name` and `Expert_Loyal` on the rfm dataframe and return it.

    Expects R_score, F_score, M_score to exist (quantile scoring applied).
    """
    df = rfm.copy()
    required = {"R_score", "F_score", "M_score"}
    if not required.issubset(df.columns):
        raise ValueError("R_score/F_score/M_score not found in rfm. Run rfm_quantile_scoring first.")

    df["Segment_Name"] = df.apply(lambda x: rfm_segment(x.R_score, x.F_score, x.M_score), axis=1)
    df["Expert_Loyal"] = df["Segment_Name"].apply(map_loyal)
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
def train_models(df):

    feature_cols = ["R_score", "F_score", "M_score", "Age", "Gender"]
    X = df[feature_cols]
    y = df["Expert_Loyal"]

    # chia 70/20/10
    X_temp, X_eval, y_temp, y_eval = train_test_split(X, y, test_size=0.10,
                                                      stratify=y, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2222,
                                                        stratify=y_temp, random_state=RANDOM_STATE)

    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    print("\n=== Decision Tree Performance ===")
    print("Train acc:", dt.score(X_train, y_train))
    print("Test acc :", dt.score(X_test, y_test))

    print("\n=== Random Forest Performance ===")
    print("Train acc:", rf.score(X_train, y_train))
    print("Test acc :", rf.score(X_test, y_test))

    joblib.dump(dt, os.path.join(OUTPUT_DIR, "dt.joblib"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf.joblib"))

    return dt, rf


# ============================================================
# MAIN FLOW
# ============================================================
def main():

    df = load_data()
    rfm = compute_rfm(df)
    rfm = rfm_quantile_scoring(rfm)
    rfm = create_expert_label(rfm)

    rfm_clustered, best_k, best_sil = run_kmeans(rfm)

    dt, rf = train_models(rfm_clustered)

    rfm_clustered.to_csv(os.path.join(OUTPUT_DIR, "rfm_scores.csv"), index=False)

    print("\n🎯 DONE. Kiểm tra thư mục outputs2/")
    print(f"KMeans best_k={best_k}, silhouette={best_sil:.3f}")


if __name__ == "__main__":
    main()