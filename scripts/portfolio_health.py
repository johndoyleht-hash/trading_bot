#!/usr/bin/env python3
# scripts/portfolio_health.py

import argparse
import sys
from pathlib import Path
import pandas as pd


def find_numbered_latest(path_glob: str):
    files = sorted(Path().glob(path_glob))
    return files[-1] if files else None


def pick_col(df: pd.DataFrame, *cands: str):
    # try exact then case-insensitive
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_trades_csv(path: Path, pair: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pair"] = pair

    # time column for ordering
    tcol = pick_col(
        df,
        "close_time", "exit_time", "timestamp", "time", "end_time", "closed_at",
        "open_time", "entry_time", "start_time",
    )
    if tcol is None:
        df["__row_i"] = range(len(df))
        tcol = "__row_i"

    # PnL column
    pnl_col = pick_col(df, "pnl", "pl", "profit", "p_l", "PnL")
    if pnl_col is None:
        raise RuntimeError(
            f"{path} is missing a recognizable PnL column "
            "(looked for: pnl/pl/profit)."
        )

    df = df.rename(columns={tcol: "t", pnl_col: "pnl"})
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    return df[["t", "pnl", "pair"]]


def profit_factor(pnls: pd.Series) -> float:
    gains = pnls[pnls > 0].sum()
    losses = pnls[pnls < 0].sum()  # negative
    return float("inf") if losses == 0 else (gains / abs(losses))


def max_drawdown(equity: pd.Series) -> float:
    peaks = equity.cummax()
    dd = (equity - peaks) / peaks
    return dd.min()  # negative (e.g., -0.07 = -7%)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--pairs", type=str, default="EURUSD,GBPUSD,USDJPY")
    ap.add_argument("--start_equity", type=float, default=25000.0)
    ap.add_argument("--folder", type=str, default="runs/live_sim")
    ap.add_argument(
        "--use_latest", action="store_true",
        help="If multiple trades_*.csv exist, pick the newest one.",
    )
    # CI thresholds (None = don't enforce)
    ap.add_argument(
        "--min_pf", type=float, default=None,
        help="Fail if portfolio PF is below this (e.g., 1.20).",
    )
    ap.add_argument(
        "--max_dd", type=float, default=None,
        help="Fail if |max drawdown| exceeds this (e.g., 0.07 = 7%).",
    )
    args = ap.parse_args()

    # search order: requested folder, then plain 'runs'
    candidates = [args.folder]
    if args.folder != "runs":
        candidates.append("runs")

    def find_latest_for_pair_year(pair: str, year: int):
        for base in candidates:
            pat = f"{base}/{pair}/{year}/trades_*.csv"
            path = (
                find_numbered_latest(pat)
                if args.use_latest
                else next(iter(sorted(Path().glob(pat))), None)
            )
            if path is not None:
                return path
        return None

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    all_trades = []

    for pair in pairs:
        path = find_latest_for_pair_year(pair, args.year)
        if path is None:
            print(
                f"⚠️  No trades file for {pair} {args.year} under "
                f"{' or '.join(candidates)}",
                file=sys.stderr,
            )
            continue
        try:
            df = load_trades_csv(path, pair)
            all_trades.append(df)
            print(f"✓ loaded {pair}: {path}")
        except Exception as e:
            print(f"⛔ failed to load {pair} from {path}: {e}", file=sys.stderr)

    if not all_trades:
        print("No trades loaded — nothing to do.", file=sys.stderr)
        sys.exit(1)

    combo = (
        pd.concat(all_trades, ignore_index=True)
        .sort_values(["t", "pair"])
        .reset_index(drop=True)
    )

    equity = args.start_equity + combo["pnl"].cumsum()
    pf = profit_factor(combo["pnl"])
    mdd = max_drawdown(equity)  # negative
    cum_ret = (equity.iloc[-1] / args.start_equity) - 1.0

    out_dir = Path(args.folder) / "portfolio" / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_out = out_dir / "combined_trades.csv"
    eq_out = out_dir / "equity_curve.csv"

    combo.assign(equity=equity).to_csv(trades_out, index=False)
    pd.DataFrame({"t": combo["t"], "equity": equity}).to_csv(eq_out, index=False)

    # summary
    print("\n=== Portfolio summary ===")
    print(f"Pairs: {', '.join(sorted(set(combo['pair'])))}")
    print(f"Trades: {len(combo):,}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Cumulative Return: {cum_ret*100:.2f}%")
    print(f"Final Equity: ${equity.iloc[-1]:,.2f}")
    print(f"Max Drawdown: {mdd*100:.2f}%")
    print(f"\nWrote combined trades -> {trades_out}")
    print(f"Wrote equity curve     -> {eq_out}")

    # CI gates
    failed = False
    if args.min_pf is not None and pf < args.min_pf:
        print(f"❌ PF {pf:.2f} is below required {args.min_pf:.2f}", file=sys.stderr)
        failed = True
    if args.max_dd is not None and abs(mdd) > args.max_dd + 1e-12:
        print(f"❌ Max DD {mdd*100:.2f}% exceeds {args.max_dd*100:.2f}%", file=sys.stderr)
        failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
