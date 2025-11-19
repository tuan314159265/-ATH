import kagglehub
from kagglehub import KaggleDatasetAdapter
from ucimlrepo import fetch_ucirepo
import pandas as pd
import os
import requests
import json
import glob

PATH = "/home/tuan/Documents/scl/251/ĐATH/new_code/raw_data"

def integration():
    '''
    UCI 
    '''
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    uci_path = os.path.join(PATH, "uci_online_retail.csv")
    if os.path.exists(uci_path):
        print("UCI Online Retail dataset already exists.")
    else:
        try:
            print("Downloading UCI dataset...")
            online_retail = fetch_ucirepo(id=352)
            df2 = online_retail.data.features
            df2.to_csv(uci_path, index=False)
            print("Integration of UCI dataset successful!")
        except Exception as e:
            print(f"An error occurred while fetching UCI dataset: {e}")

    '''
    kagle
    '''

    kaggle_path = os.path.join(PATH, "kaggle_retail_sales.csv")
    if os.path.exists(kaggle_path):
        print("Kaggle Retail Sales dataset already exists.")
    else:
        try:
            print("Downloading Kaggle dataset...")
            path_folder = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")
            csv_files = glob.glob(os.path.join(path_folder, "*.csv"))
            
            if csv_files:
                df1 = pd.read_csv(csv_files[0])
                df1.to_csv(kaggle_path, index=False)
                print("Integration of Kaggle dataset successful!")
            else:
                print("No CSV file found in Kaggle dataset.")
                
        except Exception as e:
            print(f"An error occurred while fetching Kaggle dataset: {e}")
    '''
    api
    '''
    ord_json_path = os.path.join(PATH, "ord.json")
    if os.path.exists(ord_json_path):
        print("ord.json dataset already exists.")
    else:
        try:
            print("Fetching API dataset...")
            url = "https://dummyjson.com/c/0ca3-0d75-46f0-84f9"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            with open(ord_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("Integration of API dataset successful!")
        except Exception as e:
            print(f"An error occurred while fetching dataset: {e}")

def convert_to_csv():
    ord_json_path = os.path.join(PATH, "ord.json")
    ord_csv_path = os.path.join(PATH, "ord.csv")

    if os.path.exists(ord_csv_path):
        print("ord.csv already exists.")
        return

    try:
        if not os.path.exists(ord_json_path):
            print(f"File {ord_json_path} not found.")
            return

        with open(ord_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            df3 = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "carts" in data:
                df3 = pd.DataFrame(data["carts"])
            elif "orders" in data:
                df3 = pd.DataFrame(data["orders"])
            else:
                df3 = pd.DataFrame([data])
        else:
            print("Data format not recognized.")
            return

        df3.to_csv(ord_csv_path, index=False)
        print("Conversion of ORD JSON to CSV successful!")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

def run_integrator():
    integration()
    convert_to_csv()

if __name__ == "__main__":
    run_integrator()