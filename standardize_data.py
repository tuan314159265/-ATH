import src
import pandas as pd

df_cust = src.df_cust.copy()

def standardize_customer_id(x):
    if pd.isna(x) or str(x).strip() == "":
        return "GUEST"
    x = str(x).strip()

    if not x.startswith("CUST"):
        x = "CUST" + x

    suffix = x[4:]  
    if suffix.isdigit() and len(suffix) == 3:
        x = "CUST" + "00" + suffix

    return x


df_cust["Gender"] = df_cust["Gender"].str.strip().str.lower().map({
    "male": 0,
    "female": 1
})

df_cust["Age"] = pd.to_numeric(df_cust["Age"], errors="coerce")
median_age = df_cust["Age"].median()
df_cust["Age"].fillna(median_age, inplace=True)
df_cust["Age"] = df_cust["Age"].astype(int)

df_cust["Date of Birth"] = pd.to_datetime(df_cust["Date of Birth"], errors="coerce").dt.strftime("%Y-%m-%d")
df_cust["Registration Date"] = pd.to_datetime(df_cust["Registration Date"], errors="coerce").dt.strftime("%Y-%m-%d")


df_cust = df_cust.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)

df_cust.to_csv("new_dataset/customer_dataset.csv", index=False)
