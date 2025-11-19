
-- 2. Tạo bảng Dimension: LOCATION (Địa điểm)
CREATE TABLE dim_location (
    location_id INT PRIMARY KEY,    -- Khóa chính
    country VARCHAR(100)            -- Tên quốc gia
);

-- 3. Tạo bảng Dimension: CATEGORY (Danh mục sản phẩm)
CREATE TABLE dim_category (
    category_id INT PRIMARY KEY,    -- Khóa chính
    category VARCHAR(100)           -- Tên danh mục (Electronics, Home Decor...)
);

-- 4. Tạo bảng Dimension: DATE (Thời gian)
CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,        -- Khóa chính
    date DATE,                      -- Ngày gốc (YYYY-MM-DD)
    date_key INT,                   -- Dạng số (20231201)
    day INT,
    month INT,
    month_name VARCHAR(20),
    quarter INT,
    year INT,
    is_weekend BOOLEAN              -- True nếu là T7/CN
);

-- 5. Tạo bảng Dimension: CUSTOMER (Khách hàng)
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,   -- Khóa đại diện (Surrogate Key)
    customer_id VARCHAR(50),        -- ID gốc từ hệ thống nguồn
    gender VARCHAR(20),
    age INT,
    location_id INT,                -- Khóa ngoại trỏ về dim_location
    
    -- Tạo liên kết khóa ngoại
    CONSTRAINT fk_customer_location 
        FOREIGN KEY (location_id) REFERENCES dim_location(location_id)
);

-- 6. Tạo bảng Fact: SALES (Giao dịch bán hàng)
CREATE TABLE fact_sales (
    sales_id INT PRIMARY KEY,       -- Khóa chính tự tăng
    transaction_id VARCHAR(50),     -- Mã đơn hàng gốc
    
    -- Các khóa ngoại trỏ về Dimension
    date_id INT,
    customer_key INT,
    category_id INT,
    
    -- Các chỉ số đo lường (Measures)
    quantity INT,
    unit_price DECIMAL(10, 2),
    total_amount DECIMAL(15, 2),

    -- Định nghĩa khóa ngoại
    CONSTRAINT fk_sales_date FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
    CONSTRAINT fk_sales_customer FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    CONSTRAINT fk_sales_category FOREIGN KEY (category_id) REFERENCES dim_category(category_id)
);
