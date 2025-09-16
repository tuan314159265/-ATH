import pandas as pd

df1 = pd.read_csv("online_retail.csv")
print(df1.info())
df2 = pd.read_csv("retail_sales_dataset.csv")
print(df2.info())

# Nếu cột chưa phải datetime thì chuyển đổi trước
df2["Date"] = pd.to_datetime(df2["Date"])

# Lấy ngày nhỏ nhất và lớn nhất
min_date = df2["Date"].min()
max_date = df2["Date"].max()

print("Ngày sớm nhất:", min_date)
print("Ngày muộn nhất:", max_date)
