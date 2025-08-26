# scripts/print_summary.py
import os, sys, json, importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.baseline_core as baseline

def run(pair, year):
    p = REPO_ROOT / "data" / "clean" / pair / str(year) / f"{pair}_1y_{year}_clean.csv"
    res = baseline.run_backtest(DATA_CSV_PATH=str(p), PAIR=pair)
    s = res["summary"]
    print(json.dumps({
        "pair": pair, "year": year,
        "trades": s["trades"], "pf": round(s["pf"], 2),
        "cum_ret": round(s["cum_ret"], 4),
        "final_eq": round(s["final_equity"], 2),
        "max_dd": round(s["max_dd"], 4),
        "exp": round(s["expectancy"], 2)
    }))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    run(args.pair, args.year)
