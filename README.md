# -ATH
# Retail Data Analysis Project

## Giới thiệu
Project này dùng dữ liệu bán lẻ (`online_retail.csv`, `retail_sales_dataset.csv`, `customer_dataset.csv`) để:
- Phân tích đặc điểm dữ liệu
- Trực quan hóa dữ liệu
- Sinh dữ liệu khách hàng giả định

## Yêu cầu hệ thống
- Python >= 3.10
- pip >= 22

## Cài đặt
Clone repo và cài đặt các module cần thiết:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

## Cấu trúc thư mục
├── dataset/
│   ├── online_retail.csv
│   ├── retail_sales_dataset.csv
│   └── customer_dataset.csv
│   └── data_of_customer.py
├── standardize_data.py
├── src.py
├── data_characteristic.py
├── visualization.py
├── requirements.txt
└── README.md
