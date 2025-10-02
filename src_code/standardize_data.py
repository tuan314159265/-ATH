import pandas as pd
import src  # chứa df_cust

# ------------------------------
# 1. Chuẩn hóa dữ liệu online_retail.csv
# ------------------------------

df_retail = pd.read_csv("dataset/online_retail.csv")

# --- Description ---
df_retail["Description"] = df_retail["Description"].fillna("UNKNOWN").str.strip().str.lower()

# --- Quantity ---
df_retail["Quantity"] = pd.to_numeric(df_retail["Quantity"], errors="coerce").fillna(0).astype(int)
df_retail = df_retail[df_retail["Quantity"] > 0]

# --- InvoiceDate ---
df_retail["InvoiceDate"] = pd.to_datetime(df_retail["InvoiceDate"], errors="coerce")
df_retail["InvoiceDate"] = df_retail["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

# --- UnitPrice ---
df_retail["UnitPrice"] = pd.to_numeric(df_retail["UnitPrice"], errors="coerce").round(2)
df_retail = df_retail[df_retail["UnitPrice"] > 0]

# --- CustomerID ---
def standardize_customer_id(x):
    if pd.isna(x) or str(x).strip() == "":
        return "GUEST"
    x = str(x).strip()
    if x.startswith("CUST"):
        suffix = x[4:]  
        if suffix.isdigit() and len(suffix) == 3:
            x = "CUST" + "00" + suffix
        return x
    else:
        try:
            x = str(int(float(x)))  
        except:
            return "GUEST"
        if len(x) == 3:
            x = "00" + x
        return "CUST" + x


df_retail["CustomerID"] = df_retail["CustomerID"].apply(standardize_customer_id)

# --- Country ---
df_retail["Country"] = df_retail["Country"].fillna("UNKNOWN").str.title()

# Lưu file online_retail_clean
df_retail.to_csv("new_dataset/online_retail_clean.csv", index=False)


# ------------------------------
# 2. Chuẩn hóa dữ liệu khách hàng df_cust
# ------------------------------

df_cust = src.df_cust.copy()

def standardize_customer_id_cust(x):
    if pd.isna(x) or str(x).strip() == "":
        return "GUEST"
    x = str(x).strip()
    if not x.startswith("CUST"):
        x = "CUST" + x
    suffix = x[4:]
    if suffix.isdigit() and len(suffix) == 3:
        x = "CUST" + "00" + suffix
    return x

df_cust["CustomerID"] = df_cust["CustomerID"].apply(standardize_customer_id_cust)

# --- Gender ---
df_cust["Gender"] = df_cust["Gender"].str.strip().str.lower().map({"male": 0, "female": 1})

# --- Age ---
df_cust["Age"] = pd.to_numeric(df_cust["Age"], errors="coerce")
median_age = df_cust["Age"].median()
df_cust["Age"].fillna(median_age, inplace=True)
df_cust["Age"] = df_cust["Age"].astype(int)

# --- Date columns ---
df_cust["Date of Birth"] = pd.to_datetime(df_cust["Date of Birth"], errors="coerce").dt.strftime("%Y-%m-%d")
df_cust["Registration Date"] = pd.to_datetime(df_cust["Registration Date"], errors="coerce").dt.strftime("%Y-%m-%d")

# --- Loại bỏ trùng lặp ---
df_cust = df_cust.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)

# Lưu file customer_dataset
df_cust.to_csv("new_dataset/customer_dataset.csv", index=False)


# ------------------------------
# 3. Chuẩn hóa dữ liệu retail_sales_dataset.csv
# ------------------------------


df_sales = pd.read_csv("dataset/retail_sales_dataset.csv")

# --- Transaction ID ---
df_sales["Transaction ID"] = pd.to_numeric(df_sales["Transaction ID"], errors="coerce").astype("Int64")
df_sales = df_sales.drop_duplicates(subset=["Transaction ID"])

# --- Date ---
df_sales["Date"] = pd.to_datetime(df_sales["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

# --- Customer ID ---
df_sales["Customer ID"] = df_sales["Customer ID"].apply(standardize_customer_id)

# --- Gender ---
df_sales["Gender"] = df_sales["Gender"].str.strip().str.lower().map({"male": 0, "female": 1})

# --- Age ---
df_sales["Age"] = pd.to_numeric(df_sales["Age"], errors="coerce").astype("Int64")

# --- Product Category ---
df_sales["Product Category"] = df_sales["Product Category"].fillna("unknown").str.strip().str.lower()

# --- Quantity ---
df_sales["Quantity"] = pd.to_numeric(df_sales["Quantity"], errors="coerce").astype("Int64")
df_sales = df_sales[df_sales["Quantity"] > 0]

# --- Price per Unit ---
df_sales["Price per Unit"] = pd.to_numeric(df_sales["Price per Unit"], errors="coerce").round(2)

# --- Total Amount ---
df_sales["Total Amount"] = df_sales["Quantity"] * df_sales["Price per Unit"]

# Lưu file retail_sales_clean
df_sales.to_csv("new_dataset/retail_sales_clean.csv", index=False)


print("Chuẩn hóa dữ liệu hoàn tất. 3 file đã được lưu:")
print(" - new_dataset/online_retail_clean.csv")
print(" - new_dataset/customer_dataset.csv")
print(" - new_dataset/retail_sales_clean.csv")
