#!/usr/bin/env python3
# convert_histdata_to_clean.py
# Converts HistData ASCII M1 bar file to your backtester format:
# timestamp,open,high,low,close,volume

import csv
import os
from datetime import datetime

IN_FILE  = "EURUSD_1y_2023.csv"
OUT_FILE = "EURUSD_1y_2023_clean.csv"

def sniff_delimiter(path, sample_size=4096):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(sample_size)
    # HistData M1 ASCII is usually semicolon; sometimes comma
    if ";" in sample and sample.count(";") >= sample.count(","):
        return ";"
    return ","

def parse_row(parts):
    """
    HistData formats commonly seen:
      1) "YYYYMMDD HHMMSS;Open;High;Low;Close;Volume"
      2) "YYYYMMDDHHMMSS;Open;High;Low;Close;Volume"
      (Sometimes comma instead of semicolon.)
    Returns (timestamp_str, open, high, low, close, volume) or None if unparsable.
    """
    if len(parts) < 6:
        return None

    dt_raw = parts[0].strip()

    # Split/normalize timestamp
    if " " in dt_raw:
        date_str, time_str = dt_raw.split()
    else:
        date_str, time_str = dt_raw[:8], dt_raw[8:14]

    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except Exception:
        return None

    try:
        o = float(parts[1]); h = float(parts[2]); l = float(parts[3]); c = float(parts[4])
        v = int(float(parts[5]))  # just in case it comes as float-like "0.0"
    except Exception:
        return None

    ts = dt.strftime("%Y-%m-%d %H:%M:%S")  # UTC from HistData
    return ts, o, h, l, c, v

def main():
    if not os.path.exists(IN_FILE):
        print(f"❌ Input file not found: {IN_FILE}")
        return

    delim = sniff_delimiter(IN_FILE)
    rows = []

    with open(IN_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=delim)
        for raw in reader:
            # Skip likely headers/blank lines
            if not raw or "DATE" in "".join(raw).upper() or "TIME" in "".join(raw).upper():
                continue
            rec = parse_row([p.strip() for p in raw])
            if rec:
                rows.append(rec)

    if not rows:
        print("❌ Parsed 0 rows. The file may be in an unexpected format.")
        return

    # Sort and dedupe by timestamp
    rows.sort(key=lambda r: r[0])
    seen = set()
    cleaned = []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        cleaned.append(r)

    with open(OUT_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","open","high","low","close","volume"])
        for ts, o, h, l, c, v in cleaned:
            w.writerow([ts, f"{o:.5f}", f"{h:.5f}", f"{l:.5f}", f"{c:.5f}", v])

    print(f"✅ Wrote {len(cleaned):,} rows -> {OUT_FILE}")

if __name__ == "__main__":
    main()
