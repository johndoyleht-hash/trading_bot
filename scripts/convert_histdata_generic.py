# scripts/convert_histdata_generic.py
# Merge & clean HistData minute files into one canonical CSV:
# columns: timestamp,open,high,low,close,volume  (UTC)

import os
import sys
import csv
import glob
import zipfile
from datetime import datetime
import pandas as pd

def read_histdata_csv(path):
    """
    Try to read a HistData-style CSV with flexible delimiters and headers.
    Expected columns (order may vary): date/time + O/H/L/C + volume (sometimes 0).
    Common raw formats:
      - "YYYYMMDD HH:MM:SS;Open;High;Low;Close;Volume"
      - "YYYYMMDD HHMMSS;Open;High;Low;Close;Volume"
    """
    # Detect delimiter quickly
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
    delim = ';' if ';' in first else (',' if ',' in first else None)

    # Load without header; we’ll fix columns
    df = pd.read_csv(path, header=None, sep=delim, engine='python')

    # Try to map columns
    # We expect 6 cols: dt, open, high, low, close, volume
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected columns in {path}: {df.shape}")

    # If more than 6, keep first 6
    df = df.iloc[:, :6].copy()
    df.columns = ['dt', 'open', 'high', 'low', 'close', 'volume']

    # Normalize datetime
    # Handle "YYYYMMDD HHMMSS" or "YYYYMMDD HH:MM:SS"
    def parse_dt(x):
        x = str(x).strip()
        # tolerate “YYYYMMDD HHMMSS”
        if len(x) == 15 and x[8] == ' ':
            # e.g. 20240105 134501
            return datetime.strptime(x, "%Y%m%d %H%M%S")
        # tolerate “YYYYMMDD HH:MM:SS”
        if len(x) == 17 and x[8] == ' ' and x[11] == ':' and x[14] == ':':
            return datetime.strptime(x, "%Y%m%d %H:%M:%S")
        # sometimes already “YYYY-MM-DD HH:MM:SS”
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(x, fmt)
            except Exception:
                pass
        # fallback try: pandas
        return pd.to_datetime(x, errors='coerce')

    df['timestamp'] = df['dt'].map(parse_dt)
    df = df.drop(columns=['dt'])
    df = df.dropna(subset=['timestamp'])

    # Ensure numeric
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])

    # Sort
    df = df.sort_values('timestamp')
    return df[['timestamp','open','high','low','close','volume']]

def read_histdata_zip(zip_path):
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.filename.lower().endswith('.csv'):
                with z.open(info) as f:
                    # create a temp CSV file-like via pandas
                    df = pd.read_csv(f, header=None, sep=';', engine='python')
                    # Save to a temp frame and pass through same normalizer
                    # Write to a temp .csv on disk to reuse the same function
                    tmp = zip_path + ".__tmp.csv"
                    df.to_csv(tmp, index=False, header=False)
                    try:
                        frames.append(read_histdata_csv(tmp))
                    finally:
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
    if not frames:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    return pd.concat(frames, ignore_index=True)

def convert_folder(raw_dir, year, out_csv, symbol="GBPUSD"):
    # Find files for the year (both .csv and .zip)
    candidates = []
    for ext in ("*.csv", "*.CSV", "*.zip", "*.ZIP"):
        candidates.extend(glob.glob(os.path.join(raw_dir, ext)))

    # Keep only files that mention the year (very loose filter)
    year_str = str(year)
    year_files = [p for p in candidates if year_str in os.path.basename(p)]

    if not year_files:
        print(f"[WARN] No files in {raw_dir} matched year {year}. Falling back to all files.")
        year_files = candidates

    frames = []
    for p in sorted(year_files):
        try:
            if p.lower().endswith(".zip"):
                df = read_histdata_zip(p)
            else:
                df = read_histdata_csv(p)
            frames.append(df)
            print(f"Loaded {len(df):,} rows from {os.path.basename(p)}")
        except Exception as e:
            print(f"[WARN] Skipped {p}: {e}")

    if not frames:
        print("[ERROR] No data loaded. Nothing written.")
        sys.exit(1)

    df_all = pd.concat(frames, ignore_index=True)
    # Ensure minute frequency unique & sorted
    df_all = df_all.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    # Save in canonical format
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False,
                  float_format="%.5f",
                  date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote {len(df_all):,} rows to {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Convert HistData raw files to clean CSV")
    ap.add_argument("--raw-dir", required=True, help="Folder with raw HistData files (.csv or .zip)")
    ap.add_argument("--year", required=True, type=int, help="Year to collect (e.g., 2023)")
    ap.add_argument("--out-csv", required=True, help="Output clean CSV path")
    ap.add_argument("--symbol", default="GBPUSD")
    args = ap.parse_args()

    convert_folder(args.raw_dir, args.year, args.out_csv, symbol=args.symbol)
