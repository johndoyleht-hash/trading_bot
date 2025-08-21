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
    print("\n======== MONTHLY PERFORMANCE ========")
    if not monthly:
        print("(no trades)")
        return
    first_key = sorted(monthly.keys())[0]
    yr = int(first_key.split("-")[0])

    for m in range(1, 13):
        key = f"{yr}-{m:02d}"
        rec = monthly.get(key, {
            "pl": 0.0, "trades": 0, "wins": 0, "losses": 0,
            "profits": [], "losses_list": []
        })
        profit_sum = sum(rec["profits"])
        loss_sum   = sum(rec["losses_list"])
        if loss_sum > 1e-12:
            pf_m = profit_sum / loss_sum
        elif profit_sum > 0:
            pf_m = float('inf')
        else:
            pf_m = 0.0
        win_rate = (rec["wins"] / rec["trades"] * 100) if rec["trades"] else 0.0
        ret_pct  = rec["pl"] / account_start

        print(
            f"{key} | Trades:{rec['trades']:3d} | Wins:{rec['wins']:3d} | Losses:{rec['losses']:3d} | "
            f"Win%:{win_rate:5.1f}% | P/L:${rec['pl']:9.2f} | Return:{ret_pct:6.2%} | PF:{pf_m:4.2f}"
        )

def print_funnel(funnel, examples):
    if not funnel:
        return
    print("\nSignal funnel (immediate & window):")
    keys = [
        "macd_cross", "macd+trend", "macd+trend+rsi_immediate",
        "macd+trend+rsi+atr_immediate", "rsi_met_in_window", "rsi_window_expired",
        "daily_dd_guard_active", "warmup"
    ]
    for k in keys:
        if k in funnel:
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
    data_path = f"data/clean/{pair}/{year}/{pair}_1y_{year}_clean.csv"
    
    # 4) Run using core’s public entry
    results = baseline.run_backtest(DATA_CSV_PATH=data_path, PAIR=pair)
    


    data_path = os.path.join("data", "clean", pair, str(year), f"{pair}_1y_{year}_clean.csv")
    print(f"\nPAIR={pair} YEAR={year}")
    print(f"Using DATA_CSV_PATH: {os.path.abspath(data_path)}")

    # IMPORTANT:
    # baseline_core.run_backtest must NOT print monthlies when imported.
    # It should RETURN a dict like:
    # {
    #   "monthly_results": dict,
    #   "account_start": float,
    #   "funnel": dict,
    #   "examples": {"ALL": [...]},
    #   "summary": {trades,wins,losses,wl_ratio,pf,expectancy,cum_ret,final_equity,max_dd},
    #   "trades_csv": "...optional path..."
    # }
    results = baseline.run_backtest(DATA_CSV_PATH=data_path, PAIR=pair)

    print_monthlies(results.get("monthly_results", {}), results.get("account_start", 25_000.0))

    if args.verbose:
        print_funnel(results.get("funnel", {}), results.get("examples", {}))

    s = results["summary"]
    print("\n======== BACKTEST SUMMARY (DD Guards On) ========")
    print(f"Trades: {s['trades']}  |  Wins: {s['wins']}  Losses: {s['losses']}  |  W/L Ratio: {s['wl_ratio']:.2f}")
    print(f"Profit Factor: {s['pf']:.2f}  |  Expectancy per trade: ${s['expectancy']:,.2f}")
    print(f"Cumulative Return: {s['cum_ret']:.2%}  |  Final Equity: ${s['final_equity']:,.2f}")
    print(f"Max Total Drawdown (peak-to-trough): {s['max_dd']:.2%}")

    if "trades_csv" in results and results["trades_csv"]:
        print(f"\nTrades CSV: {results['trades_csv']}")

if __name__ == "__main__":
    main()
