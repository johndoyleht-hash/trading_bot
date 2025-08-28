#!/usr/bin/env python3
# scripts/paper_sim.py
import argparse
import sys
from pathlib import Path
import inspect
import yaml  # pyyaml

# make 'src' importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.baseline_core as baseline  # noqa: E402


def run_backtest_with_config(csv_path: Path, pair: str, config_path: Path):
    """
    Load a forward YAML and call baseline.run_backtest with only supported kwargs.
    Extra keys in the YAML are ignored (warned about).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Always drop these
    for k in ("CONFIG_PATH", "PAIR", "DATA_CSV_PATH"):
        cfg.pop(k, None)

    # Introspect allowed args from run_backtest
    sig = inspect.signature(baseline.run_backtest)
    allowed = set(sig.parameters.keys())

    # Keep only supported keys; warn on ignored
    clean_cfg = {k: v for k, v in cfg.items() if k in allowed}
    dropped = sorted(set(cfg) - allowed)

    if dropped:
        print(
            f"[paper_sim] ⚠️ ignored unsupported keys for {pair} "
            f"({config_path.name}): {', '.join(dropped)}"
        )

    # Call baseline.run_backtest with sanitized config
    return baseline.run_backtest(
        DATA_CSV_PATH=str(csv_path),
        PAIR=pair,
        **clean_cfg,
    )


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

    # Try to surface the path to saved trades
    saved = None
    if isinstance(res, dict):
        for k in ("trades_csv", "trades_path", "trades_file", "path"):
            if k in res:
                saved = res[k]
                break
    if not saved:
        saved = out_dir  # best guess

    print(f"[paper_sim] saved trades: {saved}")


if __name__ == "__main__":
    main()
