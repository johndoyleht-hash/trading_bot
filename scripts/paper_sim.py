#!/usr/bin/env python3
# scripts/paper_sim.py
import argparse
from pathlib import Path
import shutil
import sys
import yaml

# Make 'src' importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.baseline_core as baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", required=True, help="path to *_forward.yaml (or baseline yaml)")
    args = ap.parse_args()

    pair = args.pair.upper()
    year = int(args.year)
    cfg_path = Path(args.config)

    # Clean data CSV (produced by your conversion step)
    csv_path = Path("data/clean") / pair / str(year) / f"{pair}_1y_{year}_clean.csv"

    # Load YAML (allow empty)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Drop purely-informational keys if present
    for k in ("CONFIG_PATH", "PAIR", "DATA_CSV_PATH"):
        cfg.pop(k, None)

    # Show what we're about to apply (helps sanity-check)
    if cfg:
        print(f"[paper_sim] overrides from {cfg_path.name}:")
        for k in sorted(cfg.keys()):
            print(f"  - {k}: {cfg[k]}")
    else:
        print(f"[paper_sim] no overrides in {cfg_path.name} (using baseline_core defaults)")

    # Run the backtest; baseline.run_backtest applies **cfg to module knobs
    res = baseline.run_backtest(DATA_CSV_PATH=str(csv_path), PAIR=pair, **cfg)

    # Discover saved trades file (baseline_core returns it in res["trades_csv"])
    saved = None
    if isinstance(res, dict):
        saved = res.get("trades_csv")

    if not saved:
        # fall back to the expected folder
        out_dir = Path("runs") / "live_sim" / pair / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = out_dir

    print(f"[paper_sim] saved trades: {saved}")

    # Maintain a deterministic pointer for portfolio scripts (latest.csv)
    try:
        trades_path = Path(saved)
        if trades_path.is_file():
            latest = trades_path.parent / "latest.csv"
            shutil.copyfile(trades_path, latest)
            print(f"[paper_sim] updated: {latest}")
    except Exception as e:
        print(f"[paper_sim] note: could not update latest.csv -> {e}")

if __name__ == "__main__":
    main()
