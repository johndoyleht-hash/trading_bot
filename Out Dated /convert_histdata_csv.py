# convert_histdata_csv.py
import csv
from datetime import datetime
import sys

in_path  = sys.argv[1] if len(sys.argv) > 1 else "EURUSD_1m_2024.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "EURUSD_1m_2024_clean.csv"

with open(in_path, newline="") as fin, open(out_path, "w", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    writer.writerow(["timestamp","open","high","low","close","volume"])  # header

    for row in reader:
        if not row or len(row) < 6:
            continue
        # Format: 0=date(YYYY.MM.DD), 1=HH:MM, 2..5=OHLC, 6=volume (often 0)
        date_s, time_s = row[0].strip(), row[1].strip()
        ts = datetime.strptime(f"{date_s} {time_s}", "%Y.%m.%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
        o, h, l, c = map(float, row[2:6])
        vol = float(row[6]) if len(row) > 6 and row[6] != "" else 0.0
        writer.writerow([ts, o, h, l, c, vol])

print(f"✅ Wrote {out_path}")
