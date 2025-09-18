import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('en_GB')

df1 = pd.read_csv("online_retail.csv")
df2 = pd.read_csv("retail_sales_dataset.csv")

def import_exitend_customer_ids():
    cust1 = df1[["CustomerID"]]
    cust2 = df2[["Customer ID", "Gender", "Age"]].rename(columns={"Customer ID": "CustomerID"})
    df_customers = pd.concat([cust1, cust2], ignore_index=True)
    df_customers.to_csv("customer_dataset.csv", index=False)

def check_data():
    df_customers = pd.read_csv("customer_dataset.csv")
    print(df_customers.head(10), df_customers.tail(10))

def generate_data():
    df_customers = pd.read_csv("customer_dataset.csv")
    df_customers = df_customers.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)
    df_customers
    # Thêm các cột mới
    df_customers["Full Name"] = ""
    df_customers["Email"] = ""
    df_customers["Phone Number"] = ""
    df_customers["Address"] = ""
    df_customers["Date of Birth"] = ""
    df_customers["Registration Date"] = ""

    for i in df_customers.index:
        full_name = fake.name()
        df_customers.at[i, "Full Name"] = full_name

        email = full_name.lower().replace(" ", ".") + str(random.randint(1, 100)) + "@gmail.com"
        df_customers.at[i, "Email"] = email

        phone_number = fake.phone_number()
        df_customers.at[i, "Phone Number"] = phone_number

        address = fake.address().replace("\n", ", ")
        df_customers.at[i, "Address"] = address

        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=70)
        df_customers.at[i, "Date of Birth"] = birth_date.strftime("%Y-%m-%d")

        registration_date = datetime.now() - timedelta(days=random.randint(0, 5*365))
        df_customers.at[i, "Registration Date"] = registration_date.strftime("%Y-%m-%d")

        if df_customers.at[i, "Age"] == "" or pd.isna(df_customers.at[i, "Age"]):
            age = (datetime.now().date() - birth_date).days // 365
            df_customers.at[i, "Age"] = age
        
        if df_customers.at[i, "Gender"] == "":
            gender = random.choice(["Male", "Female"])
            df_customers.at[i, "Gender"] = gender

    df_customers.to_csv("customer_dataset.csv", index=False)


generate_data()
check_data()