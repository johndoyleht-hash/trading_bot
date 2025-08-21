# backtest_from_csv.py
import csv
from datetime import datetime, timezone
import os
import random
import logging

# ======= Config =======
DATA_CSV_PATH     = "EURUSD_1m_2024_clean.csv"  # <-- your converted file
PRINT_SKIPS       = True
TRADE_LIMIT       = 50          # stop after N executed trades
ACCOUNT_START     = 25_000.00
RISK_PER_TRADE    = 0.01        # 1% of equity
MAX_TOTAL_DD      = 0.079       # 7.9%
MAX_DAILY_DD      = 0.039       # 3.9%
MIN_ATR           = 0.0002      # “looser” but still conservative
MACD_EPS          = 1e-6        # tiny tolerance so crosses don’t miss by a hair

# per-run output files (prevents mixing runs)
run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADES_CSV = f"trades_log_{run_id}.csv"
SKIPS_CSV  = f"skips_log_{run_id}.csv"

logging.basicConfig(
    filename=f"trading_algorithm_{run_id}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======= CSV utils =======
def write_csv_row(path, fieldnames, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ======= Indicator helpers =======
def ema_last(series, period):
    if not series:
        return 0.0
    if len(series) < period:
        return sum(series) / len(series)
    k = 2 / (period + 1)
    val = series[0]
    for x in series[1:]:
        val = x * k + val * (1 - k)
    return float(val)

def ema_series(series, span):
    if not series:
        return []
    k = 2 / (span + 1)
    out = [series[0]]
    for x in series[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    ups   = [d if d > 0 else 0.0 for d in deltas[-period:]]
    downs = [-d if d < 0 else 0.0 for d in deltas[-period:]]
    avg_gain = sum(ups) / period if sum(ups) > 0 else 0.0
    avg_loss = sum(downs) / period if sum(downs) > 0 else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def calc_true_atr(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        if n < 2: return 0.0005
        rr = [abs(closes[i] - closes[i-1]) for i in range(1, n)]
        return sum(rr[-period:]) / min(period, len(rr))
    trs = []
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        trs.append(max(hl, hc, lc))
    window = trs[-period:]
    return sum(window)/len(window) if window else 0.0005

def calc_macd(closes, fast=12, slow=26, sig=9):
    if len(closes) < slow + sig:
        return 0.0, 0.0
    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)
    macd_line_series = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_series    = ema_series(macd_line_series, sig)
    return float(macd_line_series[-1]), float(signal_series[-1])

# ======= Risk helpers =======
def stop_from_atr(price, atr, direction, mult=1.5):
    return price - atr * mult if direction == "buy" else price + atr * mult

def position_size(equity, stop_distance):
    dist = abs(stop_distance) if abs(stop_distance) > 0 else 1e-6
    risk_amount = equity * RISK_PER_TRADE
    return risk_amount / dist

# ======= Loader for your clean CSV =======
def load_ohlcv_clean(path):
    """
    Expects header: timestamp,open,high,low,close,volume
    timestamp format: 'YYYY-MM-DD HH:MM:SS'
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row["timestamp"]
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            o = float(row["open"]); h = float(row["high"])
            l = float(row["low"]);  c = float(row["close"])
            v = float(row.get("volume", 0.0))
            rows.append((dt, o, h, l, c, v))
    rows.sort(key=lambda x: x[0])
    return rows

# ======= Backtest =======
def main():
    print(f"Run ID: {run_id}")
    print("Config => strict MACD+EMA+RSI | ATR gate=0.0002 | DD: 7.9% total / 3.9% daily")
    data = load_ohlcv_clean(DATA_CSV_PATH)
    print(f"Loaded {len(data)} candles from {DATA_CSV_PATH}")

    trade_fields = ["timestamp","trade_id","direction","price","size","stop_loss","atr",
                    "p_l","equity_after","is_win","ema50","ema200","rsi","macd","signal"]
    skip_fields  = ["timestamp","reason","price","ema50","ema200","rsi","atr","macd","signal"]

    # State
    portfolio_value = ACCOUNT_START
    peak_value      = ACCOUNT_START
    daily_start     = ACCOUNT_START
    current_day     = None
    prev_macd       = None
    prev_signal     = None
    trade_count     = 0
    wins            = 0
    losses          = 0
    total_profit    = 0.0
    total_loss      = 0.0

    closes, highs, lows = [], [], []

    def apply_dd_guards():
        nonlocal portfolio_value, peak_value, daily_start
        total_dd = (peak_value - portfolio_value)/peak_value if peak_value > 0 else 0.0
        daily_dd = (daily_start - portfolio_value)/daily_start if daily_start > 0 else 0.0
        if daily_dd >= MAX_DAILY_DD:
            print(f"🚨 Daily drawdown hit: {daily_dd:.2%} — stopping day block.")
            return False
        if total_dd >= MAX_TOTAL_DD:
            print(f"🚨 Total drawdown hit: {total_dd:.2%} — stopping backtest.")
            return False
        return True

    for (dt, o, h, l, c, v) in data:
        # daily baseline reset
        day_key = dt.date()
        if current_day is None:
            current_day = day_key
            daily_start = portfolio_value
        elif day_key != current_day:
            current_day = day_key
            daily_start = portfolio_value

        closes.append(c); highs.append(h); lows.append(l)

        ema50  = ema_last(closes, 50)
        ema200 = ema_last(closes, 200)
        rsi    = calc_rsi(closes, 14)
        atr    = calc_true_atr(highs, lows, closes, 14)
        macd_line, sig_line = calc_macd(closes, 12, 26, 9)

        # initialize MACD state after enough bars
        if prev_macd is None or prev_signal is None:
            prev_macd, prev_signal = macd_line, sig_line
            write_csv_row(SKIPS_CSV, skip_fields, {
                "timestamp": dt.isoformat(sep=" "),
                "reason": "warmup/macd_state_unavailable",
                "price": f"{c:.5f}",
                "ema50": f"{ema50:.5f}",
                "ema200": f"{ema200:.5f}",
                "rsi": f"{rsi:.2f}",
                "atr": f"{atr:.5f}",
                "macd": f"{macd_line:.6f}",
                "signal": f"{sig_line:.6f}",
            })
            if PRINT_SKIPS:
                print(f"⏭️  {dt} warmup/macd_state_unavailable")
            continue

        if not apply_dd_guards():
            break

        # strict cross with EPS
        buy_cross  = (prev_macd <= prev_signal) and ((macd_line - sig_line) >  MACD_EPS)
        sell_cross = (prev_macd >= prev_signal) and ((macd_line - sig_line) < -MACD_EPS)

        direction = None
        reason = None
        if buy_cross:
            if not (ema50 > ema200):
                reason = "ema_trend_not_up_for_buy"
            elif not (rsi < 30):
                reason = "rsi_not_oversold_for_buy"
            elif not (atr > MIN_ATR):
                reason = "atr_too_low_for_buy"
            else:
                direction = "buy"
        elif sell_cross:
            if not (ema50 < ema200):
                reason = "ema_trend_not_down_for_sell"
            elif not (rsi > 70):
                reason = "rsi_not_overbought_for_sell"
            elif not (atr > MIN_ATR):
                reason = "atr_too_low_for_sell"
            else:
                direction = "sell"
        else:
            reason = "macd_no_strict_crossover"

        # advance MACD state
        prev_macd, prev_signal = macd_line, sig_line

        if direction is None:
            write_csv_row(SKIPS_CSV, skip_fields, {
                "timestamp": dt.isoformat(sep=" "),
                "reason": reason,
                "price": f"{c:.5f}",
                "ema50": f"{ema50:.5f}",
                "ema200": f"{ema200:.5f}",
                "rsi": f"{rsi:.2f}",
                "atr": f"{atr:.5f}",
                "macd": f"{macd_line:.6f}",
                "signal": f"{sig_line:.6f}",
            })
            if PRINT_SKIPS:
                print(f"⏭️  {dt} {reason}")
            continue

        # risk sizing
        sl = stop_from_atr(c, atr, direction, mult=1.5)
        size = position_size(portfolio_value, sl - c)

        # outcome model (keep your randomized payoff tied to ATR)
        is_win = random.random() > 0.5
        pl = (2 * atr * size) if is_win else (-1 * atr * size)

        portfolio_value += pl
        if portfolio_value > peak_value:
            peak_value = portfolio_value

        trade_id = trade_count + 1
        trade_count += 1
        if is_win:
            wins += 1
            total_profit += pl
        else:
            losses += 1
            total_loss += abs(pl)

        write_csv_row(TRADES_CSV, trade_fields, {
            "timestamp": dt.isoformat(sep=" "),
            "trade_id": trade_id,
            "direction": direction,
            "price": f"{c:.5f}",
            "size": f"{size:.4f}",
            "stop_loss": f"{sl:.5f}",
            "atr": f"{atr:.5f}",
            "p_l": f"{pl:.2f}",
            "equity_after": f"{portfolio_value:.2f}",
            "is_win": int(is_win),
            "ema50": f"{ema50:.5f}",
            "ema200": f"{ema200:.5f}",
            "rsi": f"{rsi:.2f}",
            "macd": f"{macd_line:.6f}",
            "signal": f"{sig_line:.6f}",
        })

        print(f"🟢 {dt}  TRADE {trade_id}: {direction.upper()}  P/L: {pl:.2f}  Equity: {portfolio_value:.2f}")

        if not apply_dd_guards():
            break

        if trade_count >= TRADE_LIMIT:
            print("Reached trade limit.")
            break

    # summary
    wl_ratio = (wins / losses) if losses > 0 else (wins if wins else 0.0)
    cum_ret  = (portfolio_value - ACCOUNT_START) / ACCOUNT_START
    print("\n======== BACKTEST SUMMARY ========")
    print(f"Trades: {trade_count} | Wins: {wins} | Losses: {losses} | W/L Ratio: {wl_ratio:.2f}")
    print(f"Cumulative Return: {cum_ret:.2%}")
    print(f"Final Equity: ${portfolio_value:,.2f}")
    print(f"CSV outputs: {TRADES_CSV}, {SKIPS_CSV}")

if __name__ == "__main__":
    main()
