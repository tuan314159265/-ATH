# Pipeline Documentation

## Overview

The data pipeline processes retail data from multiple sources through several stages:

```
Raw Data → Cleaning → Transformation → Integration → Warehouse → Analytics
```

---

## Stage 1: Data Cleaning

**File**: `cleaning/cleaning.py`

### Data Sources

| Source | Path | Records | Format |
|--------|------|---------|--------|
| Kaggle Retail | `raw_data/kaggle_retail_sales.csv` | ~500K | CSV |
| ORD API | `raw_data/ord.csv` | ~200K | CSV |
| UCI Online Retail | `raw_data/uci_online_retail.csv` | ~500K | CSV |

### Cleaning Steps

1. **Load data** with appropriate encodings
2. **Handle missing values**
   - Numerical: Forward fill or mean imputation
   - Categorical: Mode or placeholder
3. **Remove duplicates**
   - Based on transaction ID and timestamp
4. **Data type conversion**
   - Dates to datetime64
   - Amounts to float64
5. **Outlier detection**
   - Statistical methods (IQR)
   - Domain knowledge rules

### Output

Cleaned files saved to `clean_data/`:
- `clean_kaggle_sales.csv`
- `clean_ord_api.csv`
- `clean_uci_retail.csv`

---

## Stage 2: Data Transformation

**File**: `tranformation_data/tranformation.py`

### Transformations Applied

1. **Schema Standardization**
   - Rename columns to common names
   - Ensure consistent column order

2. **Feature Engineering**
   - Extract date components (year, month, day)
   - Calculate order value in standard currency
   - Create transaction flags

3. **Data Validation**
   - Check for valid date ranges
   - Validate monetary amounts (non-negative)
   - Verify required fields present

### Output

Transformed files saved to `transformed_data/`:
- `transformed_kaggle.csv`
- `transformed_ord.csv`
- `transformed_uci.csv`

---

## Stage 3: Data Integration

**File**: `integration/intergration.py`

### Integration Steps

1. **Field Mapping**
   - Customer ID harmonization
   - Transaction date alignment
   - Amount standardization

2. **Data Merging**
   - Union all sources
   - Maintain data lineage (source column)

3. **Deduplication**
   - Identify duplicate transactions across sources
   - Keep original record with earliest timestamp

### Output

Merged dataset: `final_merged_data.csv`

---

## Stage 4: ETL & Data Warehouse

**File**: `load/ETL.py`

### Database Schema

```sql
-- Dimension Tables
CREATE TABLE dim_customer (
    customer_id INT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_date DATE
);

CREATE TABLE dim_product (
    product_id INT PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(100),
    price DECIMAL(10, 2)
);

CREATE TABLE dim_date (
    date_id INT PRIMARY KEY,
    date DATE,
    year INT,
    month INT,
    day INT
);

-- Fact Table
CREATE TABLE fact_transaction (
    transaction_id INT PRIMARY KEY,
    customer_id INT,
    product_id INT,
    date_id INT,
    quantity INT,
    amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES dim_customer,
    FOREIGN KEY (product_id) REFERENCES dim_product,
    FOREIGN KEY (date_id) REFERENCES dim_date
);
```

### ETL Performance

- **Load time**: ~2-5 minutes for 1.2M records
- **Data quality**: 98%+ clean records
- **Incremental load**: Delta loading supported

---

## Stage 5: Analytics & ML Pipeline

### RFM Analysis

**File**: `modeling/rfm_analysis.py`

1. **Calculate RFM metrics**
   - Recency: Days since last purchase
   - Frequency: Number of transactions
   - Monetary: Total amount spent

2. **Preprocessing**
   - Handle outliers (IQR method)
   - Normalize features (StandardScaler)
   - Reduce skewness (BoxCox transformation)

3. **Output**: `modeling/pipeline_data/rfm_scaled.csv`

### Customer Clustering

**File**: `modeling/clustering.py`

1. **Feature preparation**
   - Use normalized RFM features
   - Handle missing values

2. **K-means clustering**
   - Elbow method: K = 3-5
   - Initialize: K-means++
   - Iterations: Until convergence

3. **Segmentation**
   - Assign customers to clusters
   - Calculate segment statistics

4. **Output**: `modeling/pipeline_data/customers_*.csv`

### Supervised Models

**File**: `modeling/supervised_models.py`

1. **Decision Tree**
   - Max depth: Auto-tuned
   - Criterion: Gini impurity
   - Cross-validation: 5-fold

2. **Random Forest**
   - Estimators: 100
   - Max features: sqrt(n_features)
   - Cross-validation: 5-fold

3. **Model evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Save best models to joblib

4. **Artifacts**: `modeling/pipeline_data/*.joblib`

---

## Running the Pipeline

### Complete Pipeline
```bash
python main.py --all
```

### Individual Steps
```bash
# RFM Analysis
python modeling/rfm_analysis.py

# Clustering
python modeling/clustering.py

# Model Training
python modeling/supervised_models.py

# Visualization
python modeling/visualization.py
```

### Interactive Dashboard
```bash
streamlit run app/app.py
```

---

## Data Quality Checks

```python
# Check completeness
df.isnull().sum()

# Check data types
df.dtypes

# Check value ranges
df.describe()

# Check for duplicates
df.duplicated().sum()

# Check data freshness
df['date_column'].max()
```

---

## Performance Optimization

1. **Caching**: Use joblib for model serialization
2. **Batch processing**: Process data in chunks if needed
3. **Indexing**: Add database indexes on foreign keys
4. **Partitioning**: Partition large tables by date

---

## Monitoring & Maintenance

- Track pipeline execution time
- Monitor data quality metrics
- Set up alerts for failures
- Schedule regular re-training
- Archive old data periodically
