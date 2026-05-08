"""
Calculate RFM (Recency, Frequency, Monetary) metrics for vãng lai (transient) customers.

Loads: `modeling/pipeline_data/full_info_customers_vanglai.csv`
Computes:
  - Recency: days since last purchase (from max date in the file)
  - Frequency: number of transactions per customer
  - Monetary: total spending per customer

Outputs: CSV with RFM scores and quintile segments to `modeling/pipeline_data/`

Run:
  python3 modeling/calculate_rfm_vanglai.py
  python3 modeling/calculate_rfm_vanglai.py --input modeling/pipeline_data/full_info_customers_vanglai.csv --out modeling/pipeline_data/rfm_vanglai.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data')
DEFAULT_INPUT = os.path.join(PIPELINE_DIR, 'full_info_customers_vanglai.csv')
DEFAULT_OUTPUT = os.path.join(PIPELINE_DIR, 'rfm_vanglai.csv')


def calculate_rfm(df, date_col='date', customer_col='customer_id', amount_col='total_amount'):
    """
    Calculate RFM metrics from transaction data.
    
    Returns DataFrame with customer_id, Recency, Frequency, Monetary
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data")
    if customer_col not in df.columns:
        raise ValueError(f"Customer column '{customer_col}' not found in data")
    if amount_col not in df.columns:
        raise ValueError(f"Amount column '{amount_col}' not found in data")

    # ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # reference date: max date in the dataset
    ref_date = df[date_col].max()
    print(f"Reference date: {ref_date}")

    # group by customer
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (ref_date - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency
        amount_col: 'sum'  # Monetary
    }).reset_index(drop=True)

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm.insert(0, customer_col, df.groupby(customer_col)[customer_col].first().values)

    return rfm


def assign_rfm_scores(rfm, n_segments=5):
    """
    Assign quintile (1-5) scores to each RFM metric.
    """
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_segments, labels=False, duplicates='drop') + 1
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_segments, labels=False, duplicates='drop') + 1
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=n_segments, labels=False, duplicates='drop') + 1

    # combine into RFM_Segment (e.g., "111", "555")
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    return rfm


def main(args):
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    print(f"📖 Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} transactions")

    try:
        rfm = calculate_rfm(df, date_col=args.date_col, customer_col=args.customer_col, amount_col=args.amount_col)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    rfm = assign_rfm_scores(rfm, n_segments=args.n_segments)

    # statistics
    print("\n📊 RFM Statistics:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
    print(f"\nSegments distribution:\n{rfm['RFM_Segment'].value_counts().sort_index()}")

    # save
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rfm.to_csv(args.out, index=False)
    print(f"\n✅ RFM metrics saved to: {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate RFM for vãng lai customers')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Input CSV with full customer info')
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT, help='Output CSV with RFM metrics')
    parser.add_argument('--date-col', type=str, default='date', help='Date column name')
    parser.add_argument('--customer-col', type=str, default='customer_id', help='Customer ID column name')
    parser.add_argument('--amount-col', type=str, default='total_amount', help='Transaction amount column name')
    parser.add_argument('--n-segments', type=int, default=5, help='Number of RFM quintiles')
    args = parser.parse_args()
    main(args)
