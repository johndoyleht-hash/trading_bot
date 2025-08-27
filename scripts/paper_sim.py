#!/usr/bin/env python3
# scripts/paper_sim.py
import argparse
import sys
from pathlib import Path

# make 'src' importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # pyyaml
import src.baseline_core as baseline  # noqa: E402


def run_backtest_with_config(csv_path: Path, pair: str, config_path: Path):
    """
    Load a forward YAML and call baseline.run_backtest with only the kwargs it expects.
    IMPORTANT: Do NOT pass CONFIG_PATH (that caused CI errors previously).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Remove keys that baseline.run_backtest doesn't accept / we never use here
    for k in ["CONFIG_PATH", "PAIR", "DATA_CSV_PATH"]:
        cfg.pop(k, None)

    # Primary call: keyword-style interface used by this repo
    try:
        return baseline.run_backtest(
            DATA_CSV_PATH=str(csv_path),
            PAIR=pair,
            **cfg,
        )
    except TypeError:
        # Fallback: some environments might expose a positional signature
        # (csv_path, pair, **cfg). Try that before giving up.
        return baseline.run_backtest(str(csv_path), pair, **cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", required=True, help="path to *_forward.yaml")
    args = ap.parse_args()

    pair = args.pair.upper()
    year = int(args.year)
    cfg_path = Path(args.config)

    # Where paper trades live
    out_dir = Path("runs") / "live_sim" / pair / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean data CSV (produced by your conversion step)
    csv_path = Path("data/clean") / pair / str(year) / f"{pair}_1y_{year}_clean.csv"

    # Run and let baseline_core handle writing trades logs
    res = run_backtest_with_config(csv_path, pair, cfg_path)

    # The baseline writes a dated trades CSV; we print where it went (for convenience)
    # If baseline returns a path or object with the path, try to surface it:
    saved = None
    for k in ("trades_csv", "trades_path", "trades_file", "path"):
        if isinstance(res, dict) and k in res:
            saved = res[k]
            break

    if not saved:
        # best effort: tell the folder where it should appear
        saved = out_dir

    print(f"[paper_sim] saved trades: {saved}")


if __name__ == "__main__":
    main()
