import src
df_cust = src.df_cust.copy()
df1 = src.df1.copy()
df2 = src.df2.copy()

# Nếu chưa có prefix "CUST" thì thêm vào
def standardize_customer_id(x):
    # Trường hợp NaN hoặc rỗng → GUEST
    if src.pd.isna(x) or str(x).strip() == "":
        return "GUEST"
    
    x = str(x).strip()  # loại bỏ khoảng trắng thừa
    # Nếu chưa có prefix CUST → thêm vào
    if not x.startswith("CUST"):
        x = "CUST" + x
    # Nếu sau CUST chỉ có 3 chữ số → thêm "00"
    suffix = x[4:]  # lấy phần sau CUST
    if suffix.isdigit() and len(suffix) == 3:
        x = "CUST" + "00" + suffix
    return x

df_cust.to_csv("new_dataset/customer_dataset.csv", index=False)