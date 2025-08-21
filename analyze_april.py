# analyze_april.py
import sys, glob, os, re
import pandas as pd
import numpy as np

# ----------------------------- Utils -----------------------------

def latest_trades_log():
    files = sorted(glob.glob("trades_log_20250818_120318.csv"))
    return files[-1] if files else None

def to_datetime_safe(s):
    # Handles both "YYYY-MM-DD HH:MM:SS" and ISO strings; returns NaT on failure
    return pd.to_datetime(s, errors="coerce", utc=True)

def summarize(label, g):
    # Profit Factor
    profits = g.loc[g["pnl"] > 0, "pnl"].sum()
    losses  = g.loc[g["pnl"] < 0, "pnl"].sum()
    pf = (profits / abs(losses)) if losses < 0 else np.nan

    return pd.Series({
        "trades": len(g),
        "win%": (g["pnl"] > 0).mean() * 100.0,
        "PF": pf,
        "avg_R": g["R"].mean() if "R" in g.columns else np.nan,
        "expectancy_$": g["pnl"].mean(),
        "gross_$": g["pnl"].sum()
    }, name=label)

def bucketize(series, edges, labels):
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)

# ----------------------------- Load -----------------------------

CSV = sys.argv[1] if len(sys.argv) > 1 else latest_trades_log()
if CSV is None or not os.path.exists(CSV):
    print("No trades_log_*.csv found. Pass a file explicitly: python3 analyze_april.py <file.csv>")
    sys.exit(1)

df = pd.read_csv(CSV)

# Your header example:
# entry_time,exit_time,direction,entry,exit,stop,target,atr_at_entry,size,p_l,r_multiple,bars_held,exit_reason,ema50,ema200,rsi,macd,signal

# ----------------------------- Normalize Columns -----------------------------

rename_map = {
    "direction": "side",
    "p_l": "pnl",
    "r_multiple": "R",
    "atr_at_entry": "atr",
}
df = df.rename(columns=rename_map)

# Coerce datetimes
if "entry_time" in df.columns: df["entry_time"] = to_datetime_safe(df["entry_time"])
if "exit_time"  in df.columns: df["exit_time"]  = to_datetime_safe(df["exit_time"])

# Coerce numerics where applicable
num_cols = ["pnl","R","atr","rsi","ema50","ema200","target","stop","bars_held","entry","exit","size","macd","signal"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Derived helpers
if {"ema50","ema200"}.issubset(df.columns):
    df["ema_spread"] = df["ema50"] - df["ema200"]
else:
    df["ema_spread"] = np.nan

# Time features from exit_time (performance is tallied at exit)
if "exit_time" in df.columns:
    et = df["exit_time"]
    df["hour"]    = et.dt.hour
    df["weekday"] = et.dt.day_name()
else:
    df["hour"] = np.nan
    df["weekday"] = np.nan

# ATR & RSI buckets (optional diagnostics)
if "atr" in df.columns:
    df["atr_bucket"] = bucketize(
        df["atr"],
        edges=[-np.inf, df["atr"].quantile(0.26), df["atr"].quantile(0.90), np.inf],
        labels=["low-mid", "mid-high", "high+"]
    )
if "rsi" in df.columns:
    df["rsi_bucket"] = bucketize(
        df["rsi"], edges=[-np.inf, 30, 40, 50, 60, 70, np.inf],
        labels=["<=30","30-40","40-50","50-60","60-70",">70"]
    )

# Side normalization
if "side" in df.columns:
    df["side"] = df["side"].str.lower().map({"buy":"long","sell":"short"}).fillna(df["side"])

# ----------------------------- Slices -----------------------------

# April 2024 window (by exit_time)
mask_april = (df["exit_time"] >= "2024-04-01") & (df["exit_time"] < "2024-05-01")
april = df.loc[mask_april].copy()
rest  = df.loc[~mask_april].copy()

# ----------------------------- Headline -----------------------------
print("\n==== HEADLINE ====")
print(summarize("April 2024", april).round(3))
print(summarize("Rest of 2024", rest).round(3))
print(summarize("All", df).round(3))

# ----------------------------- April by side -----------------------------
if "side" in df.columns:
    print("\n==== April by side ====")
    print(april.groupby("side").apply(lambda g: summarize("", g)).round(3))

# ----------------------------- April by exit_reason -----------------------------
if "exit_reason" in df.columns:
    print("\n==== April by exit_reason ====")
    by_exit = april.groupby("exit_reason").apply(lambda g: summarize("", g)).round(3)
    print(by_exit.sort_values("gross_$"))

# ----------------------------- April by exit hour -----------------------------
print("\n==== April by exit hour (UTC) ====")
if "hour" in april.columns:
    print(april.groupby("hour").apply(lambda g: summarize("", g)).fillna(0).round(3).sort_index())
else:
    print("No exit_time -> hour available.")

# ----------------------------- April by weekday -----------------------------
print("\n==== April by weekday ====")
if "weekday" in april.columns:
    # Ensure order Mon..Sun if present
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wk = april.groupby("weekday").apply(lambda g: summarize("", g)).round(3)
    wk = wk.reindex([w for w in order if w in wk.index])
    print(wk)
else:
    print("No exit_time -> weekday available.")

# ----------------------------- April by EMA spread sign -----------------------------
print("\n==== April by ema50-ema200 spread sign ====")
if "ema_spread" in april.columns:
    g = april.assign(spread_sign=np.where(april["ema_spread"] >= 0, "ema50>=ema200", "ema50<ema200"))
    print(g.groupby("spread_sign").apply(lambda x: summarize("", x)).round(3))
else:
    print("No EMA columns.")

# ----------------------------- April by RSI bucket -----------------------------
if "rsi_bucket" in april.columns:
    print("\n==== April by RSI bucket ====")
    print(april.groupby("rsi_bucket").apply(lambda g: summarize("", g)).round(3).sort_index())

# ----------------------------- April by ATR bucket -----------------------------
if "atr_bucket" in april.columns:
    print("\n==== April by ATR bucket ====")
    print(april.groupby("atr_bucket").apply(lambda g: summarize("", g)).round(3))

# ----------------------------- Holding-time buckets -----------------------------
if "bars_held" in april.columns:
    print("\n==== April by holding-time bucket (bars_held) ====")
    bins = [0, 30, 60, 120, 180, 240, 300, 10_000]
    labels = ["<=30","30-60","60-120","120-180","180-240","240-300",">300"]
    hold = april.assign(hold_bucket=pd.cut(april["bars_held"], bins=bins, labels=labels, include_lowest=True))
    print(hold.groupby("hold_bucket").apply(lambda g: summarize("", g)).round(3))

# ----------------------------- Side x hour pivot (win% and gross) -----------------------------
print("\n==== April side x hour (win% & gross $) ====")
if {"side","pnl","hour"}.issubset(april.columns):
    winrate = april.pivot_table(index="hour", columns="side", values="pnl", aggfunc=lambda x: (x>0).mean()*100)
    gross   = april.pivot_table(index="hour", columns="side", values="pnl", aggfunc=np.sum)
    winrate.columns = [f"win%_{c}" for c in winrate.columns]
    gross.columns   = [f"gross_{c}" for c in gross.columns]
    out = pd.concat([winrate, gross], axis=1).sort_index().fillna(0).round(3)
    print(out)
else:
    print("Need side, pnl, hour to build pivot.")

# ----------------------------- Worst 10 trades -----------------------------
print("\n==== April worst 10 trades by $P/L ====")
cols = ["entry_time","exit_time","side","pnl","R","exit_reason","atr","rsi","ema_spread","bars_held","target","stop","entry","exit"]
have = [c for c in cols if c in april.columns]
if len(april):
    print(april.nsmallest(10, "pnl")[have].to_string(index=False))
else:
    print("No April trades.")

