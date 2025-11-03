import numpy as np
import pandas as pd
from datetime import datetime

print("Bắt đầu gộp dữ liệu...")

online = pd.read_csv("online_retail_clean.csv", encoding="utf-8")
customers = pd.read_csv("customer_dataset.csv", encoding="utf-8")
transactions = pd.read_csv("retail_sales_clean.csv", encoding="utf-8")
ord = pd.read_json("ORD.json")

print(f"Đọc xong: {len(online):,} + {len(customers):,} + {len(transactions):,} + {len(ord):,} rows")

# ====== RENAME ======
online.rename(columns={"CustomerID":"customer_id","InvoiceDate":"date","Description":"description",
                       "UnitPrice":"unit_price","Quantity":"quantity","Country":"country"}, inplace=True)
customers.rename(columns={"CustomerID":"customer_id","Full Name":"full_name","Phone Number":"phone",
                          "Date of Birth":"dob","Registration Date":"registration_date",
                          "Gender":"gender","Age":"age"}, inplace=True)
transactions.rename(columns={"Customer ID":"customer_id","Transaction ID":"transaction_id",
                             "Price per Unit":"unit_price","Total Amount":"total_amount",
                             "Product Category":"category","Date":"date","Quantity":"quantity",
                             "Price":"price"}, inplace=True)
ord.rename(columns={"CustomerID":"customer_id","InvoiceNo":"invoice_no","StockCode":"stock_code",
                    "Description":"description","Quantity":"quantity",
                    "InvoiceDate":"invoice_date","UnitPrice":"unit_price",
                    "Country":"country"}, inplace=True)

# ====== DATE PARSE ======
def parse_date(x):
    if pd.isna(x): return pd.NaT
    x = str(x).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
                "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try: return datetime.strptime(x, fmt)
        except: pass
    return pd.NaT

date_cols = {
    "online": ['date'],
    "customers": ['dob', 'registration_date'],
    "transactions": ['date'],
    "ord": ['invoice_date'],
}

for name, cols in date_cols.items():
    df = globals()[name]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].apply(parse_date), errors='coerce')

# ====== MERGE ======
df1 = pd.merge(transactions, online,
               on=['customer_id','date','quantity','unit_price'],
               how='outer', suffixes=('_t','_o'))

df2 = pd.merge(df1, ord,
               on=['customer_id','quantity','unit_price'],
               how='outer')

final = pd.merge(df2, customers, on='customer_id', how='left')

# ====== FIX TOTAL_AMOUNT ======
final['total_amount'] = final['total_amount'].fillna(final['quantity'] * final['unit_price'])

# ====== FIX COUNTRY (CHỈNH QUAN TRỌNG NHẤT) ======
# Nếu có cả country_o và country thì ưu tiên country_o
if 'country_o' in final.columns:
    final['country'] = final['country_o'].fillna(final.get('country'))
else:
    final['country'] = final.get('country')

# Xóa country_o nếu tồn tại
final = final.drop(columns=['country_o'], errors='ignore')

# ====== CHỌN CỘT ======
cols = [
    'customer_id','full_name','gender','age','phone','dob','registration_date',
    'transaction_id','invoice_no','invoice_date','date',
    'quantity','unit_price','total_amount','price','category',
    'description','stock_code','country'
]

final = final[[c for c in cols if c in final.columns]]

# ====== XUẤT FILE ======
final = final.sort_values('date', ascending=False)
final.to_csv("transform_data.csv", index=False, encoding="utf-8")

print("✅ Xuất file transform_data.csv hoàn tất!")
