#!/usr/bin/env python3
# scripts/paper_sim.py

import argparse
import sys
from pathlib import Path
import yaml

# Make 'src' importable when run as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.baseline_core as baseline  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def pick(cfg: dict, *keys, default=None):
    """Return first present key (case-insensitive) from cfg."""
    lower = {k.lower(): k for k in cfg.keys()}
    for k in keys:
        if k in cfg:
            return cfg[k]
        lk = k.lower()
        if lk in lower:
            return cfg[lower[lk]]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="FX pair, e.g. EURUSD")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--config", required=True, help="Path to *_forward.yaml")
    args = ap.parse_args()

    pair = args.pair.upper()
    year = int(args.year)
    cfg_path = Path(args.config)

    # Clean OHLCV path for this pair/year
    csv_path = Path("data/clean") / pair / str(year) / f"{pair}_1y_{year}_clean.csv"

    # Load forward YAML (we only care about the 4 knobs)
    cfg = load_yaml(cfg_path)
    rsi_window  = pick(cfg, "RSI_WINDOW", "rsi_window")
    rsi_buy_max = pick(cfg, "RSI_BUY_MAX", "rsi_buy_max")
    rsi_sell_min = pick(cfg, "RSI_SELL_MIN", "rsi_sell_min")
    atr_p_low   = pick(cfg, "ATR_P_LOW", "atr_p_low")

    # Build kwargs only for values that exist (avoids overriding defaults with None)
    knob_kwargs = {}
    if rsi_window is not None:
        knob_kwargs["RSI_WINDOW"] = int(rsi_window)
    if rsi_buy_max is not None:
        knob_kwargs["RSI_BUY_MAX"] = float(rsi_buy_max)
    if rsi_sell_min is not None:
        knob_kwargs["RSI_SELL_MIN"] = float(rsi_sell_min)
    if atr_p_low is not None:
        knob_kwargs["ATR_P_LOW"] = float(atr_p_low)

    # Run core backtest with explicit CSV path (no 'latest.csv' anywhere)
    res = baseline.run_backtest(
        DATA_CSV_PATH=str(csv_path),
        PAIR=pair,
        **knob_kwargs,
    )

    # Surface the trades CSV path for downstream steps & humans
    saved = None
    if isinstance(res, dict):
        saved = res.get("trades_csv")
    print(f"[paper_sim] saved trades: {saved or '(engine returned no path)'}")


if __name__ == "__main__":
    main()
