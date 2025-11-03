import requests
import json

url = "https://dummyjson.com/c/0ca3-0d75-46f0-84f9"

try:
    response = requests.get(url)
    response.raise_for_status() 
    data = response.json()
    print(json.dumps(data, indent=4, ensure_ascii=False))
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(" Đã lưu dữ liệu vào output.json")
except requests.exceptions.RequestException as e:
    print(" Lỗi khi gọi API:", e)
except json.JSONDecodeError:
    print(" Phản hồi không phải JSON hợp lệ.")
