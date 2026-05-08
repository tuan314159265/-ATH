# Retail Data Analytics Pipeline

A comprehensive data analytics project that processes retail sales data from multiple sources, builds a data warehouse, and performs customer segmentation using machine learning techniques.

## Project Overview

This project implements a complete ETL (Extract, Transform, Load) pipeline for retail sales data, followed by advanced analytics including RFM analysis, customer clustering, and predictive modeling. The system processes data from three different sources and creates a star schema data warehouse in PostgreSQL.

## Features

- **Data Integration**: Combines data from Kaggle, ORD API, and UCI Online Retail datasets
- **Data Cleaning**: Handles missing values, duplicates, and data inconsistencies
- **Data Transformation**: Standardizes formats and creates unified schema
- **Data Warehousing**: Implements star schema with dimension and fact tables
- **Customer Analytics**: RFM (Recency, Frequency, Monetary) analysis
- **Machine Learning**: K-means clustering, Decision Trees, Random Forest for customer segmentation
- **Visualization**: Data exploration and model performance visualization

## Project Structure

```
├── raw_data/                    # Raw data files
│   ├── kaggle_retail_sales.csv
│   ├── ord.csv
│   ├── ord.json
│   └── uci_online_retail.csv
├── clean_data/                  # Cleaned data files
│   ├── clean_kaggle_sales.csv
│   ├── clean_ord_api.csv
│   └── clean_uci_retail.csv
├── transformed_data/            # Transformed data files
│   ├── transformed_kaggle.csv
│   ├── transformed_ord.csv
│   └── transformed_uci.csv
├── merge_data/                  # Data merging scripts
│   └── merge.py
├── load/                        # ETL and database loading
│   ├── ETL.py
│   └── demo.session.sql
├── modeling/                    # Machine learning models (MERGED)
│   ├── __init__.py             # Package initialization
│   ├── rfm_analysis.py         # RFM calculation, preprocessing, outlier removal
│   ├── clustering.py           # K-means clustering and segmentation
│   ├── supervised_models.py    # Decision tree and random forest
│   ├── visualization.py        # All visualization functions
│   ├── pipeline_data/          # Model artifacts and processed data
│   └── visualize_model/        # Model visualizations
├── visualization/              # Data visualization scripts
│   └── visualization.py
├── cleaning/                   # Data cleaning scripts
│   └── cleaning.py
├── integration/                # Data integration scripts
│   └── intergration.py
├── tranformation_data/         # Data transformation scripts
│   └── tranformation.py
├── outputs/                    # Final outputs and reports
├── main.py                     # Main entry point
└── final_merged_data.csv       # Final merged dataset
```

## Data Sources

1. **Kaggle Retail Sales**: Customer purchase data with demographics
2. **ORD API**: Order data with transaction details
3. **UCI Online Retail**: E-commerce transaction data

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Required Python packages (see requirements.txt if available)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd retail-analytics-pipeline
```

2. Install dependencies:
```bash
pip install pandas numpy sqlalchemy scikit-learn matplotlib seaborn faker
```

3. Set up PostgreSQL database:
```sql
CREATE DATABASE ord;
```

4. Update database configuration in relevant files:
   - `load/ETL.py`: Update `DB_URI` with your database credentials
   - `modeling/rfm_cal.py`: Update `DB_URI` if different

## Usage

### Complete Pipeline (Recommended)

Run the complete analytics pipeline:

```bash
python main.py
```

This will execute all steps in order:
1. Data exploration
2. ETL pipeline (load to warehouse)
3. RFM analysis and preprocessing
4. Customer clustering and segmentation
5. Supervised model training
6. Visualization generation

### Individual Components

#### 1. ETL Pipeline
```bash
python load/ETL.py
```

#### 2. RFM Analysis Pipeline
```bash
python modeling/rfm_analysis.py --action rfm
```

#### 3. Customer Clustering
```bash
python modeling/clustering.py --k 2
```

#### 4. Model Training & Comparison
```bash
python modeling/supervised_models.py
python modeling/visualization.py --action compare
```

#### 5. Generate All Visualizations
```bash
python modeling/visualization.py --action all
```

#### Visualization
```bash
python visualization/visualization.py
```

## Database Schema

The project creates a star schema data warehouse with the following tables:

### Dimension Tables
- `dim_date`: Date dimension with calendar attributes
- `dim_customer`: Customer dimension with demographics
- `dim_location`: Geographic dimension
- `dim_category`: Product category dimension

### Fact Table
- `fact_sales`: Sales transactions with foreign keys to dimensions

## Models and Outputs

### Customer Segmentation
- **RFM Analysis**: Segments customers based on recency, frequency, and monetary value
- **K-means Clustering**: Unsupervised clustering for customer groups
- **Supervised Models**: Decision trees and random forests for prediction

### Model Artifacts
- `pipeline_data/kmeans_rfm_model.joblib`: Trained K-means model
- `pipeline_data/decision_tree_cluster.joblib`: Decision tree model
- `pipeline_data/random_forest_cluster.joblib`: Random forest model
- `pipeline_data/scaler.joblib`: Data scaler
- `pipeline_data/power_transformer.joblib`: Power transformer for normalization

## Configuration

### Database Configuration
Update the `DB_URI` in the following files with your PostgreSQL credentials:
- `load/ETL.py`
- `modeling/rfm_cal.py`

Example:
```python
DB_URI = "postgresql+psycopg2://username:password@localhost:5432/ord"
```

### File Paths
Update absolute paths in scripts to match your system:
- `load/ETL.py`: `CSV_IN` path
- `merge_data/merge.py`: `TRANSFORMED_PATH` and `OUTPUT_FILE` paths

## Dependencies

- pandas
- numpy
- sqlalchemy
- scikit-learn
- matplotlib
- seaborn
- faker
- joblib

## Results and Insights

The project generates various customer segments and provides insights into:
- Customer purchase patterns
- High-value customer identification
- Sales trends and seasonality
- Product category performance
- Geographic sales distribution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Updates

### Code Reorganization (December 2025)

The modeling directory has been reorganized for better maintainability and easier code navigation:

**Before:** 11 separate files with scattered functionality
- `rfm_cal.py`, `calculate_rfm_vanglai.py`, `extract_vanglai.py`
- `k_mean.py`
- `dec_tree+ran_for.py`
- `visua_kmean.py`, `visual_after_boxcox.py`, `visualize_post_analysis.py`, `compare_and_visualize.py`
- `lookup.py`

**After:** 4 consolidated modules
- `rfm_analysis.py` - All RFM-related calculations and preprocessing
- `clustering.py` - K-means clustering and customer segmentation
- `supervised_models.py` - Decision tree and random forest models
- `visualization.py` - All plotting and visualization functions

**Benefits:**
- Easier code navigation and maintenance
- Reduced file count from 11 to 4 core modules
- Better function organization and documentation
- Simplified import structure
- Updated main.py with integrated pipeline execution

All original functionality is preserved with improved structure and documentation.