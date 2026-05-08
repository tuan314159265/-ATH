"""
Extract rows from final_merged_data.csv where customer_id == 0
and write them to modeling/pipeline_data/full_info_customers_vanglai.csv.

Usage:
  python3 modeling/extract_vanglai.py
  python3 modeling/extract_vanglai.py --master /path/to/final_merged_data.csv --out modeling/pipeline_data/full_info_customers_vanglai.csv
"""
import os
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MASTER = os.path.join(PROJECT_ROOT, 'final_merged_data.csv')
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, 'modeling', 'pipeline_data', 'full_info_customers_vanglai.csv')


def extract_vanglai(master_path=None, out_path=None, chunksize=100000):
    master = master_path or DEFAULT_MASTER
    out = out_path or DEFAULT_OUTPUT

    if not os.path.exists(master):
        print(f"❌ Master file not found: {master}")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)

    written = 0
    # Use chunks to avoid memory spikes
    for i, chunk in enumerate(pd.read_csv(master, chunksize=chunksize)):
        # detect id column name
        id_col = None
        for c in ('customer_id', 'CustomerID', 'id'):
            if c in chunk.columns:
                id_col = c
                break
        if id_col is None:
            print("❌ Không tìm thấy cột ID trong file master. Mong đợi 'customer_id' hoặc 'id'.")
            return 0

        sel = chunk[chunk[id_col] == 0]
        if not sel.empty:
            # write header for first chunk, append later
            if written == 0:
                sel.to_csv(out, index=False, mode='w')
            else:
                sel.to_csv(out, index=False, mode='a', header=False)
            written += len(sel)
        print(f"Processed chunk {i+1}: found {len(sel)} rows with {id_col}==0")

    print(f"✅ Done. Total rows written: {written} -> {out}")
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master', type=str, default=None, help='Path to final_merged_data.csv')
    parser.add_argument('--out', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    extract_vanglai(master_path=args.master, out_path=args.out)
