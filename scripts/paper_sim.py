#!/usr/bin/env python3
# scripts/paper_sim.py

import argparse
import sys
from pathlib import Path
from shutil import copyfile
import yaml

# Make 'src' importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.baseline_core as baseline  # noqa: E402


def run_backtest_with_config(csv_path: Path, pair: str, config_path: Path):
    """
    Load a forward YAML and call baseline.run_backtest with only supported kwargs.
    This protects us from 'unexpected keyword argument' errors when YAML contains
    extra knobs that baseline.run_backtest doesn't accept.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Drop meta keys we never forward
    for k in ("CONFIG_PATH", "PAIR", "DATA_CSV_PATH"):
        cfg.pop(k, None)

    # Only keep kwargs that baseline.run_backtest actually accepts
    import inspect
    allowed = set(inspect.signature(baseline.run_backtest).parameters.keys())
    clean_cfg = {k: v for k, v in cfg.items() if k in allowed}

    # Run
    return baseline.run_backtest(DATA_CSV_PATH=str(csv_path), PAIR=pair, **clean_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", required=True, help="path to *_forward.yaml")
    args = ap.parse_args()

    pair = args.pair.upper()
    year = int(args.year)
    cfg_path = Path(args.config)

    # Input OHLCV
    csv_path = Path("data/clean") / pair / str(year) / f"{pair}_1y_{year}_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing clean data: {csv_path}")

    # Output dir for trades
    out_dir = Path("runs") / "live_sim" / pair / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run backtest
    res = run_backtest_with_config(csv_path, pair, cfg_path)

    # Try to locate the path of the trades file baseline wrote
    saved = None
    if isinstance(res, dict):
        for k in ("trades_csv", "trades_path", "trades_file", "path"):
            if k in res:
                saved = res[k]
                break

    if not saved:
        # best effort: pick the newest trades_* in out_dir
        candidates = sorted(out_dir.glob("trades_*.csv"))
        if candidates:
            saved = str(candidates[-1])

    print(f"[paper_sim] saved trades: {saved if saved else out_dir}")

    # Write a stable alias so CI/health checks always read the same file
    try:
        if saved:
            saved_path = Path(saved)
        else:
            saved_path = sorted(out_dir.glob("trades_*.csv"))[-1]
        stable = out_dir / "latest.csv"
        copyfile(saved_path, stable)
        print(f"[paper_sim] wrote stable alias: {stable}")
    except Exception as e:
        print(f"[paper_sim] ⚠ could not write latest.csv alias: {e}")


if __name__ == "__main__":
    main()
