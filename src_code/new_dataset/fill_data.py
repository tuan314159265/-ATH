from faker import Faker
import pandas as pd
import numpy as np

transform_data = pd.read_csv("transform_data.csv", low_memory=False)
fake = Faker('en_US')

# ========= ĐẢM BẢO CÁC CỘT TỒN TẠI ==========
required_cols = ["full_name", "phone", "gender", "dob", "country", "category"]
for col in required_cols:
    if col not in transform_data.columns:
        transform_data[col] = np.nan   # tạo cột nếu thiếu

transform_data['customer_id'] = None  # xoá sạch
n = len(transform_data)

# Tạo customer id dạng CUST000001, CUST000002, ...
transform_data['customer_id'] = [
    f"CUST{str(i).zfill(6)}" for i in range(1, n + 1)
]

# ========= FULL NAME ==========
if transform_data['full_name'].isna().any():
    transform_data['full_name'] = transform_data['full_name'].astype("object")
    transform_data.loc[transform_data['full_name'].isna(), 'full_name'] = [
        fake.name() for _ in range(transform_data['full_name'].isna().sum())
    ]

# ========= PHONE ==========
if transform_data['phone'].isna().any():
    transform_data['phone'] = transform_data['phone'].astype("object")
    transform_data.loc[transform_data['phone'].isna(), 'phone'] = [
        fake.phone_number().replace('x', '') 
        for _ in range(transform_data['phone'].isna().sum())
    ]

# ========= GENDER ==========
if transform_data['gender'].isna().any() or (transform_data['gender'].astype(str).any()) not in ['0','1']:
    transform_data['gender'] = transform_data['gender'].astype('object')
    transform_data.loc[transform_data['gender'].isna(), 'gender'] = np.random.choice(
        ['0', '1'],
        size=transform_data['gender'].isna().sum()
    )

# ========= DATE OF BIRTH ==========
if transform_data['dob'].isna().any():
    transform_data['dob'] = transform_data['dob'].astype("object")
    start = pd.Timestamp('1955-01-01')
    end   = pd.Timestamp('2007-01-01')

    fake_dobs = [
        fake.date_between(start_date=start, end_date=end)
        for _ in range(transform_data['dob'].isna().sum())
    ]

    transform_data.loc[transform_data['dob'].isna(), 'dob'] = fake_dobs

countries = ['United Kingdom', 'Germany', 'France', 'USA', 'Australia']
transform_data['country'] = transform_data['country'].astype("object")

if transform_data['country'].isna().any():
    transform_data.loc[transform_data['country'].isna(), 'country'] = np.random.choice(
        countries, size=transform_data['country'].isna().sum()
    )

# ========= CATEGORY ==========
categories = [
    'Electronics','Clothing','Home','Beauty','Toys',
    'Sports','Books','Food','Jewelry','Other'
]
transform_data['category'] = transform_data['category'].astype("object")

if transform_data['category'].isna().any():
    transform_data.loc[transform_data['category'].isna(), 'category'] = np.random.choice(
        categories, size=transform_data['category'].isna().sum()
    )

if transform_data['age'].isna().any():
    transform_data['age'] = transform_data['age'].astype("float")
    current_year = pd.Timestamp.now().year

    def infer_age(dob):
        if pd.isna(dob):
            return np.nan
        try:
            year = pd.to_datetime(dob).year
            return current_year - year
        except:
            return np.nan

    transform_data['age'] = transform_data.apply(
        lambda row: infer_age(row['dob']) if pd.isna(row['age']) else row['age'],
        axis=1
    )
def fake_stock_code():
    return str(np.random.randint(0, 100000)).zfill(5)

if transform_data['stock_code'].isna().any():
    transform_data['stock_code'] = transform_data['stock_code'].astype("string")
    transform_data.loc[transform_data['stock_code'].isna(), 'stock_code'] = [
        fake_stock_code() for _ in range(transform_data['stock_code'].isna().sum())
    ]

print(f"Filled {transform_data.isna().sum().sum()} missing cells with realistic data!")
transform_data.to_csv("transform_data_filled.csv", index=False)
print("✅ Đã lưu file transform_data_filled.csv")