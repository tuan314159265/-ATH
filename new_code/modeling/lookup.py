import pandas as pd
import os
import warnings
import argparse
import glob

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MASTER = os.path.join(PROJECT_ROOT, "final_merged_data.csv")
DEFAULT_INPUT_DIRS = [
    os.path.join(PROJECT_ROOT, "modeling", "pipeline_data"),
    os.path.join(PROJECT_ROOT, "outputs", "customers_by_segment"),
    os.path.join(PROJECT_ROOT, "outputs", "customers_vang_lai"),
    os.getcwd()
]


def resolve_master_path(candidate=None):
    if candidate and os.path.exists(candidate):
        return candidate
    if os.path.exists(DEFAULT_MASTER):
        return DEFAULT_MASTER
    # try candidates relative to cwd
    for d in DEFAULT_INPUT_DIRS:
        p = os.path.join(d, os.path.basename(DEFAULT_MASTER))
        if os.path.exists(p):
            return p
    return None


def find_input_files(input_dir=None, pattern='customers_*.csv'):
    dirs = []
    if input_dir:
        dirs.append(input_dir)
    dirs.extend(DEFAULT_INPUT_DIRS)
    seen = set()
    files = []
    for d in dirs:
        if not d:
            continue
        if not os.path.exists(d):
            continue
        candidates = glob.glob(os.path.join(d, pattern))
        for c in candidates:
            if c not in seen:
                seen.add(c)
                files.append(c)
    return files


def enrich_data_from_ids(master_path, files_to_process):
    print("🚀 Starting Data Enrichment (Merge IDs with Master Data)...")

    if not os.path.exists(master_path):
        print(f"❌ Lỗi: Không tìm thấy file gốc '{master_path}'.")
        return

    df_master = pd.read_csv(master_path)
    print(f"✅ Loaded Master Data: {len(df_master)} rows")

    # 2. Xác định tên cột ID trong file Master
    id_col ='customer_id' if 'customer_id' in df_master.columns else ('id' if 'id' in df_master.columns else None)
    if id_col is None:
        print("❌ Lỗi: Không tìm thấy cột ID trong file Master.")
        return

    print(f"🔑 Using join key: '{id_col}'\n")

    for input_path in files_to_process:
        input_name = os.path.basename(input_path)
        try:
            df_ids = pd.read_csv(input_path)
        except Exception as e:
            print(f"⚠️ Không thể đọc '{input_path}': {e}")
            continue

        if id_col not in df_ids.columns:
            # try common id names
            possible = [c for c in df_ids.columns if c.lower() in ('customerid', 'id', 'cust_id')]
            if possible:
                df_ids = df_ids.rename(columns={possible[0]: id_col})
            else:
                print(f"⚠️ File '{input_name}' không có cột ID hợp lệ, bỏ qua.")
                continue

        df_merged = pd.merge(df_ids, df_master, on=id_col, how='inner')

        output_path = os.path.join(os.path.dirname(input_path), f"full_info_{input_name}")
        df_merged.to_csv(output_path, index=False)

        print(f"✅ Xử lý '{input_name}':")
        print(f"   - Input IDs: {len(df_ids)}")
        print(f"   - Matched Rows: {len(df_merged)} (Full Info)")
        print(f"   - Saved to: {output_path}")
        print("-" * 30)

    print("\n🏁 Hoàn tất quá trình tách và làm giàu dữ liệu.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enrich ID files by joining with master dataset')
    parser.add_argument('--master', type=str, default=None, help='Path to master CSV (final_merged_data.csv)')
    parser.add_argument('--input-dir', type=str, default=None, help='Directory to look for customer ID files')
    parser.add_argument('--process-all', action='store_true', help='Process all customer_*.csv files in input-dir or default dirs')
    parser.add_argument('--files', type=str, nargs='*', help='Specific input files to process')
    args = parser.parse_args()

    master_path = resolve_master_path(args.master)
    if master_path is None:
        print("❌ Không tìm thấy file master. Vui lòng chỉ định --master <path> hoặc đặt 'final_merged_data.csv' vào project root.")
        raise SystemExit(1)

    if args.files:
        files = [os.path.abspath(f) for f in args.files]
    elif args.process_all:
        files = find_input_files(args.input_dir)
    else:
        # default known filenames (try several directories)
        candidate_names = ["customers_vanglai.csv", "customers_vang_lai.csv", "customers_diamond.csv", "customers_standard.csv"]
        files = []
        dirs = [args.input_dir] + DEFAULT_INPUT_DIRS if args.input_dir else DEFAULT_INPUT_DIRS
        for d in dirs:
            if not d:
                continue
            for name in candidate_names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    files.append(p)
        # if still empty, try pattern search
        if not files:
            files = find_input_files(args.input_dir)

    if not files:
        print("⚠️ Không tìm thấy file ID nào để xử lý. Kiểm tra --input-dir hoặc dùng --process-all.")
        raise SystemExit(0)

    enrich_data_from_ids(master_path, files)