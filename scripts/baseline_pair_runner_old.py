#!/usr/bin/env python3
import argparse
import os
import sys

# Resolve project root (go up from scripts/)
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

# Import the SINGLE source-of-truth baseline logic
# (Edit src/baseline_core.py to change behavior everywhere)
from src import baseline_core as baseline

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline_core on any pair/year using same logic."
    )
    parser.add_argument("--pair", default="EURUSD", help="e.g. EURUSD, GBPUSD (default: EURUSD)")
    parser.add_argument("--year", type=int, default=2023, help="e.g. 2023, 2024 (default: 2023)")
    args = parser.parse_args()

    pair = args.pair.upper()
    year = args.year

    data_path = os.path.join(ROOT, "data", "clean", pair, str(year), f"{pair}_1y_{year}_clean.csv")
    if not os.path.exists(data_path):
        print(f"❌ Data file not found:\n  {data_path}")
        sys.exit(1)

    # Force the baseline to read this file; everything else stays identical
    baseline.DATA_CSV_PATH = data_path

    # (Optional) ensure runs/{PAIR}/{YEAR}/ exists for logs
    runs_dir = os.path.join(ROOT, "runs", pair, str(year))
    os.makedirs(runs_dir, exist_ok=True)

    print(f"PAIR={pair} YEAR={year}")
    print(f"Using DATA_CSV_PATH: {baseline.DATA_CSV_PATH}")
    baseline.main()

if __name__ == "__main__":
    main()
