import pandas as pd
import numpy as np

df = pd.read_csv("new_dataset/transform_data.csv")

df['date_key'] = df['date'].str.replace('-', '').astype(int)


dim_date = pd.DataFrame()
dim_date['date'] = pd.to_datetime(df['date']).dropna().unique()
dim_date['date_key'] = dim_date['date'].dt.strftime("%Y%m%d").astype(int)
dim_date['day'] = dim_date['date'].dt.day
dim_date['month'] = dim_date['date'].dt.month
dim_date['year'] = dim_date['date'].dt.year
dim_date['quarter'] = dim_date['date'].dt.quarter
dim_date['is_weekend'] = dim_date['date'].dt.weekday >= 5


dim_product = df[['stock_code','description','category']].drop_duplicates()
dim_product.insert(0, 'product_key', np.arange(1, len(dim_product)+1))


dim_location = df[['country']].drop_duplicates()
dim_location.insert(0, 'country_key', np.arange(1, len(dim_location)+1))


dim_customer = df[['customer_id','full_name','gender','age','dob','registration_date','phone','country']]
dim_customer = dim_customer.drop_duplicates()

dim_customer = dim_customer.merge(dim_location, on='country')
dim_customer.insert(0, 'customer_key', np.arange(1, len(dim_customer)+1))

dim_customer = df[['customer_id','full_name','gender','age','dob','registration_date','phone','country']]
dim_customer = dim_customer.drop_duplicates()

dim_customer = dim_customer.merge(dim_location, on='country')
dim_customer.insert(0, 'customer_key', np.arange(1, len(dim_customer)+1))

fact = df[['customer_id','stock_code','date_key','invoice_no','transaction_id',
           'quantity','unit_price','total_amount']]

fact = fact.merge(dim_customer[['customer_id','customer_key']], on='customer_id')
fact = fact.merge(dim_product[['stock_code','product_key']], on='stock_code')

fact = fact[['customer_key','product_key','date_key',
             'invoice_no','transaction_id','quantity','unit_price','total_amount']]

fact.insert(0, 'sales_key', np.arange(1, len(fact)+1))
