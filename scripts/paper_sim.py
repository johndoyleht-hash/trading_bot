#!/usr/bin/env python3
import argparse, os, datetime as dt
from pathlib import Path
import importlib

# Import the same baseline core
import src.baseline_core as baseline  # adjust if your module path differs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", required=True, help="YAML config to use")
    ap.add_argument("--data", help="Optional override of CSV path")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.data:
        csv_path = Path(args.data)
    else:
        csv_path = repo / "data" / "clean" / args.pair / str(args.year) / f"{args.pair}_1y_{args.year}_clean.csv"
    assert csv_path.exists(), f"Missing data: {csv_path}"

    # reimport to avoid sticky state across multiple runs
    importlib.reload(baseline)

    res = baseline.run_backtest(
        DATA_CSV_PATH=str(csv_path),
        PAIR=args.pair,
        CONFIG_PATH=args.config  # baseline_core already supports this style in your runner
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo / "runs" / "live_sim" / args.pair / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save trades log if present in result, else dump summary
    trades_csv = res.get("trades_csv")
    if trades_csv and Path(trades_csv).exists():
        # Copy the produced trades CSV under live_sim (keeps your engine’s native CSV)
        dst = out_dir / f"trades_{stamp}.csv"
        dst.write_bytes(Path(trades_csv).read_bytes())
        print(f"[paper_sim] saved trades: {dst}")
    else:
        # fallback: just write a tiny summary
        s = res.get("summary", {})
        (out_dir / f"summary_{stamp}.json").write_text(str(s))
        print(f"[paper_sim] saved summary json (no trades csv emitted).")

if __name__ == "__main__":
    main()
