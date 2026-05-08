"""
Main Entry Point for Customer Analytics Pipeline

This script demonstrates the complete customer analytics workflow:
1. Data loading and exploration
2. RFM analysis and preprocessing
3. Customer clustering
4. Supervised model training
5. Visualization and reporting
"""

import pandas as pd
import os
import sys

# Add modeling package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modeling'))

# Import merged modules
from modeling.rfm_analysis import load_data as load_rfm_data, calculate_rfm, remove_outliers_rfm, preprocess_rfm_only, find_best_k
from modeling.clustering import clustering_and_split_id_only
from modeling.supervised_models import main as train_supervised_models
from modeling.visualization import main as run_visualizations

# Import ETL module
import load.ETL as etl


def explore_raw_data():
    """Explore and display basic information about raw data sources"""
    print("=" * 60)
    print("🔍 EXPLORING RAW DATA SOURCES")
    print("=" * 60)

    # Load raw data files
    try:
        df1 = pd.read_csv("raw_data/kaggle_retail_sales.csv")
        df2 = pd.read_csv("raw_data/ord.csv")
        df3 = pd.read_csv("raw_data/uci_online_retail.csv")

        print('Kaggle Retail Sales:')
        print(f'  - Shape: {df1.shape}')
        print(f'  - Columns: {df1.columns.tolist()}')
        print(f'  - Sample: {len(df1)} rows\n')

        print('ORD API Data:')
        print(f'  - Shape: {df2.shape}')
        print(f'  - Columns: {df2.columns.tolist()}')
        print(f'  - Sample: {len(df2)} rows\n')

        print('UCI Online Retail:')
        print(f'  - Shape: {df3.shape}')
        print(f'  - Columns: {df3.columns.tolist()}')
        print(f'  - Sample: {len(df3)} rows\n')

    except FileNotFoundError as e:
        print(f"❌ Error loading raw data: {e}")
        return False

    return True


def run_rfm_pipeline():
    """Run the complete RFM analysis pipeline"""
    print("=" * 60)
    print("📊 RFM ANALYSIS PIPELINE")
    print("=" * 60)

    try:
        # 1. Load & Calculate RFM
        df_raw = load_rfm_data()
        df_full = calculate_rfm(df_raw)

        # 2. Clean data (outlier removal)
        df_clean = remove_outliers_rfm(df_full)

        # 3. Preprocess (transformations)
        rfm_scaled = preprocess_rfm_only(df_clean)

        # 4. Find optimal K
        best_k = find_best_k(rfm_scaled)

        print(f"✅ RFM pipeline completed. Suggested K: {best_k}")
        return True

    except Exception as e:
        print(f"❌ RFM pipeline failed: {e}")
        return False


def run_clustering_pipeline():
    """Run customer clustering and segmentation"""
    print("=" * 60)
    print("🎯 CUSTOMER CLUSTERING PIPELINE")
    print("=" * 60)

    try:
        clustering_and_split_id_only()
        print("✅ Clustering pipeline completed")
        return True

    except Exception as e:
        print(f"❌ Clustering pipeline failed: {e}")
        return False


def run_model_training():
    """Train supervised models for customer segmentation prediction"""
    print("=" * 60)
    print("🤖 SUPERVISED MODEL TRAINING")
    print("=" * 60)

    try:
        # Import argparse to create args object
        import argparse
        args = argparse.Namespace(
            labeled='modeling/pipeline_data/data_labeled.csv',
            target='cluster',
            save_dir=None,
            dt_max_depth=None,
            rf_estimators=100,
            cv=5
        )
        train_supervised_models(args)
        print("✅ Model training completed")
        return True

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False


def run_visualization():
    """Generate all visualizations"""
    print("=" * 60)
    print("📈 GENERATING VISUALIZATIONS")
    print("=" * 60)

    try:
        # Import argparse to create args object
        import argparse
        args = argparse.Namespace(
            action='all',
            labeled=None,
            save_dir=None
        )
        run_visualizations(args)
        print("✅ Visualization generation completed")
        return True

    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")
        return False


def run_etl_pipeline():
    """Run the ETL pipeline to load data into warehouse"""
    print("=" * 60)
    print("🏭 ETL PIPELINE")
    print("=" * 60)

    try:
        etl.etl()
        print("✅ ETL pipeline completed")
        return True

    except Exception as e:
        print(f"❌ ETL pipeline failed: {e}")
        return False


def main():
    """Main execution function"""
    print("🚀 CUSTOMER ANALYTICS PIPELINE")
    print("=" * 80)

    # Step 1: Explore raw data
    if not explore_raw_data():
        print("❌ Failed to load raw data. Exiting.")
        return

    # Step 2: Run ETL (load data to warehouse)
    print("\n" + "="*80)
    if not run_etl_pipeline():
        print("❌ ETL pipeline failed. Some downstream processes may not work.")

    # Step 3: RFM Analysis
    print("\n" + "="*80)
    if not run_rfm_pipeline():
        print("❌ RFM analysis failed. Exiting.")
        return

    # Step 4: Customer Clustering
    print("\n" + "="*80)
    if not run_clustering_pipeline():
        print("❌ Clustering failed. Exiting.")
        return

    # Step 5: Supervised Model Training
    print("\n" + "="*80)
    if not run_model_training():
        print("❌ Model training failed. Continuing with visualization...")

    # Step 6: Generate Visualizations
    print("\n" + "="*80)
    if not run_visualization():
        print("❌ Visualization generation failed.")

    print("\n" + "="*80)
    print("🎉 PIPELINE EXECUTION COMPLETED!")
    print("="*80)
    print("\n📁 Check the following directories for outputs:")
    print("  - modeling/pipeline_data/: Model artifacts and processed data")
    print("  - modeling/visualize_model/: Charts and visualizations")
    print("  - outputs/model_comparison/: Model comparison results")


if __name__ == "__main__":
    main()