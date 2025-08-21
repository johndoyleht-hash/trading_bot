# fast_backtest_from_csv.py
# Full-year backtest, no trade cap, streaming indicators, next-bar entries,
# SL/TP/time exits, DAILY & TOTAL drawdown guards, and an RSI ±2-bar window after MACD cross
# (no look-ahead: if RSI confirms later, we enter on the next bar open when it does).

import csv
from datetime import datetime
import os
from collections import Counter

# ========= Speed / Logging =========
FAST_MODE      = True      # no per-bar printing
LOG_TRADES     = True      # write completed trades to CSV
PROGRESS_EVERY = 50_000    # bars between progress pings

# ========= Data / Strategy Config =========
DATA_CSV_PATH   = "EURUSD_1m_2024_clean.csv"   # columns: timestamp,open,high,low,close,volume
ACCOUNT_START   = 25_000.00
RISK_PER_TRADE  = 0.01       # 1% risk per trade
MIN_ATR         = 0.00017    # ATR gate (EURUSD M1 tuned)
MACD_EPS        = 1e-6       # tiny tolerance for cross
ATR_MULT_SL     = 1.5        # stop multiple (R)
ATR_MULT_TP     = 3.0        # target multiple (TP = 3R)
MAX_BARS_HOLD   = 1440       # time-based exit (~1 day on M1)

# Entry filter tuning (current thresholds)
RSI_BUY_MAX     = 29.0
RSI_SELL_MIN    = 6.0

# NEW: allow RSI to confirm within N bars AFTER the MACD cross (no look-ahead)
RSI_WINDOW_AHEAD = 2         # bars after cross in which RSI may confirm

# ========= Risk Guards =========
MAX_DAILY_DD    = 0.04       # 4% from start-of-day equity
MAX_TOTAL_DD    = 0.07       # 4% from peak equity

# ========= Per-run outputs =========
run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADES_CSV   = f"trades_log_{run_id}.csv"   # one row per completed trade

# ========= CSV utils =========
def write_csv_row(path, fieldnames, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ========= Loader =========
def load_ohlcv_clean(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            o = float(row["open"]); h = float(row["high"])
            l = float(row["low"]);  c = float(row["close"])
            v = float(row.get("volume", 0.0))
            rows.append((dt, o, h, l, c, v))
    rows.sort(key=lambda x: x[0])
    return rows

# ========= Streaming (O(1)) Indicators =========
class StreamIndicators:
    def __init__(self):
        self.rsi_p = 14
        self.atr_p = 14
        self.ema50_p, self.ema200_p = 50, 200
        self.macd_fast, self.macd_slow, self.macd_sig = 12, 26, 9
        self.k_rsi  = 2 / (self.rsi_p + 1)
        self.k_atr  = 2 / (self.atr_p + 1)
        self.k_50   = 2 / (self.ema50_p + 1)
        self.k_200  = 2 / (self.ema200_p + 1)
        self.k_f    = 2 / (self.macd_fast + 1)
        self.k_s    = 2 / (self.macd_slow + 1)
        self.k_sig  = 2 / (self.macd_sig  + 1)
        self.prev_close = None
        self.avg_gain = None
        self.avg_loss = None
        self.rsi = 50.0
        self.atr = None
        self.ema50 = None
        self.ema200 = None
        self.ema_fast_val = None
        self.ema_slow_val = None
        self.macd = None
        self.signal = None

    def update(self, high, low, close):
        # EMA50 / EMA200
        self.ema50  = close if self.ema50  is None else (close * self.k_50  + self.ema50  * (1 - self.k_50))
        self.ema200 = close if self.ema200 is None else (close * self.k_200 + self.ema200 * (1 - self.k_200))

        # RSI (EMA-style smoothing with Wilder-like behavior)
        if self.prev_close is None:
            self.prev_close = close
        else:
            delta = close - self.prev_close
            gain  = delta if delta > 0 else 0.0
            loss  = -delta if delta < 0 else 0.0
            self.avg_gain = gain if self.avg_gain is None else (gain * self.k_rsi + self.avg_gain * (1 - self.k_rsi))
            self.avg_loss = loss if self.avg_loss is None else (loss * self.k_rsi + self.avg_loss * (1 - self.k_rsi))
            if not self.avg_loss:
                self.rsi = 100.0
            else:
                rs = (self.avg_gain or 0.0) / self.avg_loss
                self.rsi = 100.0 - (100.0 / (1.0 + rs))
            self.prev_close = close

        # ATR (true range, EMA-smoothed)
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.atr = tr if self.atr is None else (tr * self.k_atr + self.atr * (1 - self.k_atr))

        # MACD fast/slow EMAs + signal
        self.ema_fast_val = close if self.ema_fast_val is None else (close * self.k_f + self.ema_fast_val * (1 - self.k_f))
        self.ema_slow_val = close if self.ema_slow_val is None else (close * self.k_s + self.ema_slow_val * (1 - self.k_s))
        macd_now = self.ema_fast_val - self.ema_slow_val
        self.signal = macd_now if self.signal is None else (macd_now * self.k_sig + self.signal * (1 - self.k_sig))
        self.macd = macd_now

        return {
            "ema50": self.ema50,
            "ema200": self.ema200,
            "rsi": self.rsi,
            "atr": self.atr,
            "macd": self.macd,
            "signal": self.signal,
        }

# ========= Utility =========
def position_size(equity, entry, stop):
    dist = abs(entry - stop)
    if dist <= 0:
        dist = 1e-6
    risk_amt = equity * RISK_PER_TRADE
    return risk_amt / dist

# ========= Backtest =========
def main():
    print(f"Run ID: {run_id}")
    print("Config => strict MACD+EMA+RSI (RSI may confirm within +2 bars), ATR gate, next-bar entry, TP/SL/time exits")
    print(f"Risk Guards => DAILY DD {MAX_DAILY_DD:.1%} | TOTAL DD {MAX_TOTAL_DD:.1%} (peak-to-trough)")

    data = load_ohlcv_clean(DATA_CSV_PATH)
    total_bars = len(data)
    print(f"Loaded {total_bars} candles from {DATA_CSV_PATH}")

    trade_fields = [
        "entry_time","exit_time","direction","entry","exit","stop","target","atr_at_entry",
        "size","p_l","r_multiple","bars_held","exit_reason","ema50","ema200","rsi","macd","signal"
    ]

    ind = StreamIndicators()
    prev_macd = None
    prev_signal = None

    portfolio = ACCOUNT_START
    peak_equity = ACCOUNT_START

    # Daily guard state
    current_day = None
    daily_start_equity = ACCOUNT_START
    daily_blocked = False  # no new entries after hit; resets next day

    # Perf aggregates
    n_trades = n_wins = n_losses = 0
    gross_profit = 0.0
    gross_loss   = 0.0
    max_drawdown = 0.0

    # Signal funnel counters
    funnel = Counter()
    examples = {"ALL": []}  # first few timestamps where *immediate* all conditions passed

    # NEW: track a pending signal after a MACD cross awaiting RSI confirmation (within +N bars)
    pending = None  # dict with: {'dir': 'buy'/'sell', 'expire': int, 'cross_idx': int}

    i = 0
    while i < total_bars:
        dt, o, h, l, c, v = data[i]

        # progress ping
        if FAST_MODE and (i % PROGRESS_EVERY == 0) and i > 0:
            pct = (i / total_bars) * 100
            print(f"Processed {i:,}/{total_bars:,} bars ({pct:.2f}%)")

        # Day rollover: reset daily start and unblock
        if current_day is None:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False
        elif dt.date() != current_day:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False

        # Update indicators on this bar
        vals = ind.update(h, l, c)
        ema50, ema200 = vals["ema50"], vals["ema200"]
        rsi, atr = vals["rsi"], vals["atr"]
        macd_line, sig_line = vals["macd"], vals["signal"]

        # Establish prior MACD state
        if prev_macd is None or prev_signal is None:
            prev_macd, prev_signal = macd_line, sig_line
            funnel["warmup"] += 1
            i += 1
            continue

        # Enforce drawdown guards BEFORE considering a new entry
        total_dd = (peak_equity - portfolio) / peak_equity if peak_equity > 0 else 0.0
        daily_dd = (daily_start_equity - portfolio) / daily_start_equity if daily_start_equity > 0 else 0.0
        if total_dd >= MAX_TOTAL_DD:
            print("⛔ Total DD limit reached — stopping backtest.")
            break
        if daily_blocked or daily_dd >= MAX_DAILY_DD:
            daily_blocked = True
            funnel["daily_dd_guard_active"] += 1
            prev_macd, prev_signal = macd_line, sig_line
            i += 1
            continue

        # Strict MACD cross on this bar
        buy_cross  = (prev_macd <= prev_signal) and ((macd_line - sig_line) >  MACD_EPS)
        sell_cross = (prev_macd >= prev_signal) and ((macd_line - sig_line) < -MACD_EPS)

        # ----- Funnel accounting (immediate) -----
        if buy_cross or sell_cross:
            funnel["macd_cross"] += 1
            trend_ok_now = (ema50 > ema200) if buy_cross else (ema50 < ema200)
            if trend_ok_now:
                funnel["macd+trend"] += 1
                rsi_immediate_ok = (rsi < RSI_BUY_MAX) if buy_cross else (rsi > RSI_SELL_MIN)
                if rsi_immediate_ok:
                    funnel["macd+trend+rsi_immediate"] += 1
                    if atr > MIN_ATR:
                        funnel["macd+trend+rsi+atr_immediate"] += 1
                        if len(examples["ALL"]) < 5:
                            examples["ALL"].append(
                                (dt.isoformat(sep=" "), float(rsi), float(atr), float(ema50), float(ema200))
                            )

        # ----- Determine entry signal (immediate or pending) -----
        direction = None

        # 1) New cross: create/replace pending if RSI not yet ok (but trend must match now)
        if (buy_cross or sell_cross):
            dir_now = "buy" if buy_cross else "sell"
            trend_ok_now = (ema50 > ema200) if dir_now == "buy" else (ema50 < ema200)
            if trend_ok_now:
                rsi_ok_now = (rsi < RSI_BUY_MAX) if dir_now == "buy" else (rsi > RSI_SELL_MIN)
                if (rsi_ok_now and atr > MIN_ATR):
                    # Immediate entry on this bar -> execute at next bar open
                    direction = dir_now
                    pending = None  # clear any previous pending
                else:
                    # Start/refresh pending signal waiting for RSI to confirm within +N bars
                    pending = {
                        "dir": dir_now,
                        "expire": i + RSI_WINDOW_AHEAD,
                        "cross_idx": i
                    }
            # else: ignore cross if trend not aligned now

        # 2) If no immediate entry, check pending window (RSI must confirm on this bar)
        if direction is None and pending is not None:
            if i > pending["expire"]:
                funnel["rsi_window_expired"] += 1
                pending = None
            else:
                # On this bar, does RSI meet threshold WITH trend & ATR?
                want_buy = (pending["dir"] == "buy")
                trend_ok_now = (ema50 > ema200) if want_buy else (ema50 < ema200)
                rsi_ok_now = (rsi < RSI_BUY_MAX) if want_buy else (rsi > RSI_SELL_MIN)
                if trend_ok_now and rsi_ok_now and (atr > MIN_ATR):
                    direction = pending["dir"]   # confirmed; enter next bar open
                    funnel["rsi_met_in_window"] += 1
                    pending = None  # consume it

        # Advance MACD state for next bar
        prev_macd, prev_signal = macd_line, sig_line

        # If no trade, move on
        if direction is None:
            i += 1
            continue

        # --------- EXECUTION ----------
        # Enter on NEXT bar open
        if i + 1 >= total_bars:
            break
        dt_entry, o1, h1, l1, c1, v1 = data[i + 1]
        entry = o1
        stop  = entry - atr * ATR_MULT_SL if direction == "buy" else entry + atr * ATR_MULT_SL
        target= entry + atr * ATR_MULT_TP if direction == "buy" else entry - atr * ATR_MULT_TP
        size = position_size(portfolio, entry, stop)

        # Walk forward until exit
        bars_held = 0
        exit_reason = None
        exit_price = None
        j = i + 1

        while j < total_bars:
            dt_j, oj, hj, lj, cj, vj = data[j]
            bars_held += 1

            if direction == "buy":
                hit_stop  = (lj <= stop)
                hit_tp    = (hj >= target)
                if hit_stop and hit_tp:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_stop:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_tp:
                    exit_price = target; exit_reason = "target"; break
            else:
                hit_stop  = (hj >= stop)
                hit_tp    = (lj <= target)
                if hit_stop and hit_tp:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_stop:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_tp:
                    exit_price = target; exit_reason = "target"; break

            if bars_held >= MAX_BARS_HOLD:
                exit_price = cj; exit_reason = "time_exit"; break

            j += 1

        if exit_price is None:
            dt_exit, _, _, _, last_c, _ = data[-1]
            exit_price = last_c
            exit_time = dt_exit
            exit_reason = "data_end"
        else:
            exit_time = data[j][0]

        # P&L and equity/drawdowns
        if direction == "buy":
            pl = (exit_price - entry) * size
            r_mult = (exit_price - entry) / (entry - stop)
        else:
            pl = (entry - exit_price) * size
            r_mult = (entry - exit_price) / (stop - entry)

        portfolio += pl
        peak_equity = max(peak_equity, portfolio)  # update peak for total DD

        # Update daily guard AFTER exits too
        daily_dd = (daily_start_equity - portfolio) / daily_start_equity if daily_start_equity > 0 else 0.0
        if daily_dd >= MAX_DAILY_DD:
            daily_blocked = True  # no more entries today

        # Track max drawdown (peak-to-trough)
        total_dd_now = (peak_equity - portfolio) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, total_dd_now)

        # Trade stats
        if pl >= 0:
            n_wins += 1
            gross_profit += pl
        else:
            n_losses += 1
            gross_loss += -pl
        n_trades += 1

        if LOG_TRADES:
            write_csv_row(
                TRADES_CSV,
                trade_fields,
                {
                    "entry_time": dt_entry.isoformat(sep=" "),
                    "exit_time":  exit_time.isoformat(sep=" "),
                    "direction":  direction,
                    "entry":      f"{entry:.5f}",
                    "exit":       f"{exit_price:.5f}",
                    "stop":       f"{stop:.5f}",
                    "target":     f"{target:.5f}",
                    "atr_at_entry": f"{atr:.5f}",
                    "size":       f"{size:.4f}",
                    "p_l":        f"{pl:.2f}",
                    "r_multiple": f"{r_mult:.2f}",
                    "bars_held":  bars_held,
                    "exit_reason": exit_reason,
                    "ema50":      f"{ema50:.5f}",
                    "ema200":     f"{ema200:.5f}",
                    "rsi":        f"{rsi:.2f}",
                    "macd":       f"{macd_line:.6f}",
                    "signal":     f"{sig_line:.6f}",
                }
            )

        # Continue scanning from the bar AFTER exit
        i = j + 1
        continue

    # ======== Signal Funnel Summary ========
    print("\nSignal funnel (immediate & window):")
    print(f" MACD strict crosses:                 {funnel['macd_cross']:,}")
    print(f" + EMA trend (same bar):              {funnel['macd+trend']:,}")
    print(f" + RSI immediate (same bar):          {funnel['macd+trend+rsi_immediate']:,}")
    print(f" + ATR gate immediate:                {funnel['macd+trend+rsi+atr_immediate']:,}")
    print(f" RSI met within +{RSI_WINDOW_AHEAD} bars (entries via window): {funnel['rsi_met_in_window']:,}")
    print(f" RSI window expired (no confirm):     {funnel['rsi_window_expired']:,}")
    # Common skip buckets from earlier versions (for context)
    for key in ["daily_dd_guard_active","warmup"]:
        if funnel[key]:
            print(f" {key:35} {funnel[key]:,}")

    if examples['ALL']:
        print("\nFirst qualifying bars (all immediate filters passed):")
        for ts, rsi_v, atr_v, e50, e200 in examples["ALL"]:
            print(f"  {ts} | RSI={rsi_v:.1f} ATR={atr_v:.5f} EMA50={e50:.5f} EMA200={e200:.5f}")

    # ======== Backtest Summary ========
    cum_ret = (portfolio - ACCOUNT_START) / ACCOUNT_START
    wl = (n_wins / n_losses) if n_losses > 0 else (float(n_wins) if n_wins else 0.0)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    expectancy = ((gross_profit - gross_loss) / max(n_trades,1))

    print("\n======== BACKTEST SUMMARY (DD Guards On) ========")
    print(f"Trades: {n_trades}  |  Wins: {n_wins}  Losses: {n_losses}  |  W/L Ratio: {wl:.2f}")
    print(f"Profit Factor: {pf:.2f}  |  Expectancy per trade: ${expectancy:,.2f}")
    print(f"Cumulative Return: {cum_ret:.2%}  |  Final Equity: ${portfolio:,.2f}")
    print(f"Max Total Drawdown (peak-to-trough): {max_drawdown:.2%}")
    print(f"Daily guard: blocks entries once daily loss ≥ {MAX_DAILY_DD:.1%} from day start; resets next day.")
    print(f"Total guard: stops backtest once loss ≥ {MAX_TOTAL_DD:.1%} from peak equity.")

    if LOG_TRADES:
        print(f"\nTrades CSV: {TRADES_CSV}")

if __name__ == "__main__":
    main()
