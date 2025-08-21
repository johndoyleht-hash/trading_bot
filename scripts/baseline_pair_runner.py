#!/usr/bin/env python3
import argparse
import os  
import sys

# add repo root to import path so we can import src/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src import baseline_core as baseline  # must expose run_backtest()

import yaml

def apply_config_to_module(mod, cfg: dict):
    """
    Push flat YAML key/values into the baseline_core module.
    Converts lists to sets for EXCLUDE_* keys, because core expects sets.
    """
    for k, v in cfg.items():
        if k in ("EXCLUDE_UTC_HOURS", "EXCLUDE_WEEKDAYS"):
            # YAML usually has lists; core uses sets
            v = set(v)
        setattr(mod, k, v)

def print_monthlies(monthly, account_start):
    """
    Pretty monthly table from results['monthly_results'].
    """
    if not monthly:
        print("⚠️ No monthly results were collected.")
        return

    first_key = sorted(monthly.keys())[0]
    year = int(first_key.split("-")[0])
    ordered_months = [f"{year}-{m:02d}" for m in range(1, 13)]

    print("\n======== MONTHLY PERFORMANCE ========")
    print(f"{'Month':7} | {'Trd':>3} | {'Win':>3} | {'Los':>3} | {'Win%':>5} | {'P/L':>11} | {'Ret%':>6} | {'PF':>4}")

    for mkey in ordered_months:
        m = monthly.get(mkey, {
            "pl": 0.0, "trades": 0, "wins": 0, "losses": 0,
            "profits": [], "losses_list": []
        })
        trades = m["trades"]; wins = m["wins"]; losses = m["losses"]; pl = m["pl"]
        win_rate = (wins / trades * 100.0) if trades else 0.0
        profit_sum = sum(m["profits"])
        loss_sum   = sum(m["losses_list"])
        if loss_sum > 1e-12:
            pf = profit_sum / loss_sum
        elif profit_sum > 0:
            pf = float('inf')
        else:
            pf = 0.0
        ret_pct = (pl / account_start) * 100.0

        print(
            f"{mkey:7} | {trades:3d} | {wins:3d} | {losses:3d} | "
            f"{win_rate:5.1f} | ${pl:10.2f} | {ret_pct:6.2f}% | {pf:4.2f}"
        )


def print_funnel(funnel, examples):
    """
    Print the signal-funnel counts and a few qualifying examples.
    """
    if not funnel:
        return

    print("\nSignal funnel (immediate & window):")
    keys = [
        "macd_cross",
        "macd+trend",
        "macd+trend+rsi_immediate",
        "macd+trend+rsi+atr_immediate",
        "rsi_met_in_window",
        "rsi_window_expired",
        "daily_dd_guard_active",
        "warmup",
    ]
    for k in keys:
        if funnel.get(k):
            print(f" {k:35} {funnel[k]:,}")

    if examples and examples.get("ALL"):
        print("\nFirst qualifying bars (all immediate filters passed):")
        for ts, rsi_v, atr_v, e50, e200 in examples["ALL"][:5]:
            print(f"  {ts} | RSI={rsi_v:.1f} ATR={atr_v:.5f} EMA50={e50:.5f} EMA200={e200:.5f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="Symbol (e.g., EURUSD, GBPUSD)")
    ap.add_argument("--year", required=True, type=int, help="Year (e.g., 2023)")
    ap.add_argument("--verbose", action="store_true", help="Show funnel + examples")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/baseline_2023_EURUSD.yaml",
        help="Path to YAML config file to load before running"
    )
    
    args = ap.parse_args()

    pair = args.pair.upper()
    year = args.year

    pair = args.pair.upper()
    year = int(args.year)
    
    # 1) Load YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    
    # 2) Apply into core
    apply_config_to_module(baseline, cfg)
        
    # 3) Build/override data path
    data_path = os.path.join("data", "clean", pair, str(year), f"{pair}_1y_{year}_clean.csv")
    print(f"\nPAIR={pair} YEAR={year}")
    print(f"Using DATA_CSV_PATH: {os.path.abspath(data_path)}")

    # 4) Run using core’s public entry
    results = baseline.run_backtest(DATA_CSV_PATH=data_path, PAIR=pair)

    # 5) Monthly table
    print_monthlies(results.get("monthly_results", {}), results.get("account_start", 25_000.0))

    # 6) Optional funnel if requested
    if args.verbose:
        print_funnel(results.get("funnel", {}), results.get("examples", {}))

    # 7) Concise backtest summary (matches the frozen script format)
    s = results["summary"]
    print("\n======== BACKTEST SUMMARY (DD Guards On) ========")
    print(f"Trades: {s['trades']}  |  Wins: {s['wins']}  Losses: {s['losses']}  |  W/L Ratio: {s['wl_ratio']:.2f}")
    print(f"Profit Factor: {s['pf']:.2f}  |  Expectancy per trade: ${s['expectancy']:,.2f}")
    print(f"Cumulative Return: {s['cum_ret']:.2%}  |  Final Equity: ${s['final_equity']:,.2f}")
    print(f"Max Total Drawdown (peak-to-trough): {s['max_dd']:.2%}")

    # 8) Show path to trades CSV if provided
    if results.get("trades_csv"):
        print(f"\nTrades CSV: {results['trades_csv']}")
if __name__ == "__main__":
    main()
