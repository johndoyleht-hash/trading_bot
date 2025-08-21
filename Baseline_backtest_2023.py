# edits_backtest.py
# EURUSD M1 2023 backtest:
# - MACD cross + recent EMA trend
# - RSI confirm (window)
# - ATR rolling percentile gate
# - Next-bar entry, fixed SL/TP, optional time exit + breakeven
# - DAILY and TOTAL DD guards
# - Monthly performance breakdown (by EXIT month)

import csv
import os
import bisect
from collections import deque, Counter, defaultdict
from datetime import datetime

# ========= Speed / Logging =========
FAST_MODE      = True
LOG_TRADES     = True
PROGRESS_EVERY = 50_000

# ========= Data / Strategy Config =========
DATA_CSV_PATH   = "EURUSD_1y_2023_clean.csv"   # change per run
ACCOUNT_START   = 25_000.00
RISK_PER_TRADE  = 0.005
MACD_EPS        = 1e-6

# Exits
ATR_MULT_SL     = 1.5
ATR_MULT_TP     = 2.1
MAX_BARS_HOLD   = 240

# Entry filters
RSI_BUY_MAX     = 33.0
RSI_SELL_MIN    = 67.0

# ======== RSI confirmation window ========
RSI_WINDOW_AHEAD = 6

# ======== ATR gate via rolling percentile ========
ATR_BAND_LOOKBACK = 1000
ATR_P_LOW         = 0.28
ATR_P_HIGH        = 0.90
ATR_MIN_WARMUP    = 200

# ======== EMA “recent” trend window ========
EMA_TREND_RECENT  = 6

# ======== Breakeven ========
USE_BREAKEVEN     = True
BE_MULT           = 0.90
BE_MIN_BARS_HELD  = 15   # only if you actually check this before arming BE

# ======== Session filter ========
USE_SESSION_FILTER  = True
SESSION_START_HHMM  = 600
SESSION_END_HHMM    = 2100
TRADE_WEEKDAYS_ONLY = True

# Extra session robustness
EXCLUDE_UTC_HOURS  = {11, 12}   # you saw 13 also help; you can use {11,12,13}
EXCLUDE_WEEKDAYS   = set()
# April/robustness combo guard (used later in the entry logic)
DISABLE_LONG_COMBO  = True       # skip LONG when ema50>=ema200 AND RSI<=cutoff
RSI_LONG_BAD_CUTOFF = 30.0


# ======== April-only tightening ========
USE_APRIL_ATR_TIGHTENING = True
APRIL_ATR_P_LOW = 0.32

# (Optional) dynamic wider SL — leave False unless you’ve hooked it up everywhere
DYN_SL_ENABLE          = False
ATR_PCNT_WIDE_SL_FROM  = 0.70
ATR_MULT_SL_BASE       = 1.5
ATR_MULT_SL_WIDE       = 2.0

# ========= Risk Guards =========
MAX_DAILY_DD    = 0.038
MAX_TOTAL_DD    = 0.07

def atr_band_ok_calendar(dt, atr, rp):
    """
    Calendar-aware ATR band check.
    Uses tighter lower bound in April if enabled; otherwise baseline band.
    """
    if USE_APRIL_ATR_TIGHTENING and dt.month == 4:
        return rp.band_ok(atr, APRIL_ATR_P_LOW, ATR_P_HIGH)
    return rp.band_ok(atr, ATR_P_LOW, ATR_P_HIGH)

# ========= Per-run outputs =========
run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
TRADES_CSV = f"trades_log_{run_id}.csv"

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

# ========= Rolling Percentile (O(log N) update) =========
class RollingPercentile:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = deque()
        self.sorted = []

    def __len__(self):
        return len(self.buf)

    def update(self, x):
        self.buf.append(x)
        bisect.insort(self.sorted, x)
        if len(self.buf) > self.maxlen:
            old = self.buf.popleft()
            idx = bisect.bisect_left(self.sorted, old)
            if idx < len(self.sorted) and self.sorted[idx] == old:
                self.sorted.pop(idx)

    def quantile(self, q):
        if not self.sorted:
            return None
        n = len(self.sorted)
        if n == 1:
            return self.sorted[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return self.sorted[lo] * (1 - frac) + self.sorted[hi] * frac

    def band_ok(self, x, qlow=0.3, qhigh=0.9):
        # Permissive during warmup to avoid starving early trades
        if len(self) < ATR_MIN_WARMUP:
            return True
        lo = self.quantile(qlow)
        hi = self.quantile(qhigh)
        if lo is None or hi is None:
            return True
        return lo <= x <= hi

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
        # EMAs
        self.ema50  = close if self.ema50  is None else (close * self.k_50  + self.ema50  * (1 - self.k_50))
        self.ema200 = close if self.ema200 is None else (close * self.k_200 + self.ema200 * (1 - self.k_200))

        # RSI
        if self.prev_close is None:
            self.prev_close = close
        else:
            d = close - self.prev_close
            gain = d if d > 0 else 0.0
            loss = -d if d < 0 else 0.0
            self.avg_gain = gain if self.avg_gain is None else (gain * self.k_rsi + self.avg_gain * (1 - self.k_rsi))
            self.avg_loss = loss if self.avg_loss is None else (loss * self.k_rsi + self.avg_loss * (1 - self.k_rsi))
            if not self.avg_loss:
                self.rsi = 100.0
            else:
                rs = (self.avg_gain or 0.0) / self.avg_loss
                self.rsi = 100.0 - (100.0 / (1.0 + rs))
            self.prev_close = close

        # ATR
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.atr = tr if self.atr is None else (tr * self.k_atr + self.atr * (1 - self.k_atr))

        # MACD
        self.ema_fast_val = close if self.ema_fast_val is None else (close * self.k_f + self.ema_fast_val * (1 - self.k_f))
        self.ema_slow_val = close if self.ema_slow_val is None else (close * self.k_s + self.ema_slow_val * (1 - self.k_s))
        macd_now = self.ema_fast_val - self.ema_slow_val
        self.signal = macd_now if self.signal is None else (macd_now * self.k_sig + self.signal * (1 - self.k_sig))
        self.macd = macd_now

        return {"ema50": self.ema50, "ema200": self.ema200, "rsi": self.rsi,
                "atr": self.atr, "macd": self.macd, "signal": self.signal}

# ========= Utility =========
def position_size(equity, entry, stop):
    dist = abs(entry - stop)
    if dist <= 0:
        dist = 1e-6
    risk_amt = equity * RISK_PER_TRADE
    return risk_amt / dist

def trend_recent_ok(ema50_hist, ema200_hist, lookback, side):
    n = min(lookback, len(ema50_hist))
    if n == 0:
        return False
    for k in range(1, n+1):
        e50 = ema50_hist[-k]; e200 = ema200_hist[-k]
        if e50 is None or e200 is None:
            continue
        if side == "buy" and (e50 > e200):
            return True
        if side == "sell" and (e50 < e200):
            return True
    return False

# ========= Backtest =========
def main():
    print(f"Run ID: {run_id}")
    print("Config => strict MACD+EMA+RSI (RSI windowed), ATR percentile band, next-bar entry, TP/SL/time exits")
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
    daily_blocked = False

    # Perf aggregates
    n_trades = n_wins = n_losses = 0
    gross_profit = 0.0
    gross_loss   = 0.0
    max_drawdown = 0.0

    # Signal funnel counters
    funnel = Counter()
    examples = {"ALL": []}

    # Pending signal after MACD cross awaiting RSI confirmation
    pending = None  # {'dir': 'buy'/'sell', 'expire': int, 'cross_idx': int}

    # Histories for recent checks and ATR band
    ema50_hist, ema200_hist = [], []
    atr_rp = RollingPercentile(ATR_BAND_LOOKBACK)

    # Monthly accumulator (by EXIT month)
    monthly = defaultdict(lambda: {
        "pl": 0.0, "trades": 0, "wins": 0, "losses": 0,
        "profits": [], "losses_list": []
    })

    i = 0
    while i < total_bars:
        dt, o, h, l, c, v = data[i]

        # ----- Session / calendar filter -----
        if USE_SESSION_FILTER:
            hhmm = dt.hour * 100 + dt.minute
            if not (SESSION_START_HHMM <= hhmm <= SESSION_END_HHMM):
                if ind.macd is not None and ind.signal is not None:
                    prev_macd, prev_signal = ind.macd, ind.signal
                i += 1
                continue

        if TRADE_WEEKDAYS_ONLY and dt.weekday() >= 5:
            if ind.macd is not None and ind.signal is not None:
                prev_macd, prev_signal = ind.macd, ind.signal
            i += 1
            continue

        # --- extra session robustness (skip fragile windows) ---
        if dt.hour in EXCLUDE_UTC_HOURS:
            if ind.macd is not None and ind.signal is not None:
                prev_macd, prev_signal = ind.macd, ind.signal
            i += 1
            continue

        if dt.weekday() in EXCLUDE_WEEKDAYS:
            if ind.macd is not None and ind.signal is not None:
                prev_macd, prev_signal = ind.macd, ind.signal
            i += 1
            continue

        # Progress ping
        if FAST_MODE and (i % PROGRESS_EVERY == 0) and i > 0:
            pct = (i / total_bars) * 100
            print(f"Processed {i:,}/{total_bars:,} bars ({pct:.2f}%)")

        # Day rollover
        if current_day is None:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False
        elif dt.date() != current_day:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False

        # Update indicators
        vals = ind.update(h, l, c)
        ema50, ema200 = vals["ema50"], vals["ema200"]
        rsi, atr = vals["rsi"], vals["atr"]
        macd_line, sig_line = vals["macd"], vals["signal"]

        # Update histories / ATR window
        ema50_hist.append(ema50)
        ema200_hist.append(ema200)
        atr_rp.update(atr if atr is not None else 0.0)

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
            dir_now = "buy" if buy_cross else "sell"
            trend_ok_now = trend_recent_ok(ema50_hist, ema200_hist, EMA_TREND_RECENT, dir_now)
            if trend_ok_now:
                funnel["macd+trend"] += 1
                rsi_immediate_ok = (rsi < RSI_BUY_MAX) if dir_now == "buy" else (rsi > RSI_SELL_MIN)
                if rsi_immediate_ok:
                    funnel["macd+trend+rsi_immediate"] += 1
                    atr_band_ok_now = atr_band_ok_calendar(dt, atr, atr_rp)
                    if atr_band_ok_now:
                        funnel["macd+trend+rsi+atr_immediate"] += 1
                        if len(examples["ALL"]) < 5:
                            examples["ALL"].append(
                                (dt.isoformat(sep=" "), float(rsi), float(atr), float(ema50), float(ema200))
                            )

        # ----- Determine entry signal (immediate or pending) -----
        direction = None

        # 1) New cross: create/replace pending if RSI not yet ok (trend must be OK recently)
        if (buy_cross or sell_cross):
            dir_now = "buy" if buy_cross else "sell"
            trend_ok_recent = trend_recent_ok(ema50_hist, ema200_hist, EMA_TREND_RECENT, dir_now)
            if trend_ok_recent:
                rsi_ok_now = (rsi < RSI_BUY_MAX) if dir_now == "buy" else (rsi > RSI_SELL_MIN)
                atr_band_ok_now = atr_band_ok_calendar(dt, atr, atr_rp)
                if (rsi_ok_now and atr_band_ok_now):
                    # === Step B guard: skip longs when ema50>=ema200 AND RSI<=cutoff ===
                    if DISABLE_LONG_COMBO and dir_now == "buy":
                        if (ema50 is not None and ema200 is not None) and (ema50 >= ema200) and (rsi <= RSI_LONG_BAD_CUTOFF):
                            direction = None
                            pending = None
                        else:
                            direction = dir_now
                            pending = None
                    else:
                        direction = dir_now
                        pending = None
                else:
                    # Start/refresh pending signal waiting for RSI to confirm within +N bars
                    pending = {
                        "dir": dir_now,
                        "expire": i + RSI_WINDOW_AHEAD,
                        "cross_idx": i
                    }
                
            # else: ignore cross if trend not recently aligned

        # 2) If no immediate entry, check pending window (RSI must confirm on this bar)
        if direction is None and pending is not None:
            if i > pending["expire"]:
                funnel["rsi_window_expired"] += 1
                pending = None
            else:
                want_buy = (pending["dir"] == "buy")
                trend_ok_recent = trend_recent_ok(ema50_hist, ema200_hist, EMA_TREND_RECENT, pending["dir"])
                rsi_ok_now = (rsi < RSI_BUY_MAX) if want_buy else (rsi > RSI_SELL_MIN)
                atr_band_ok_now = atr_band_ok_calendar(dt, atr, atr_rp)
                if trend_ok_recent and rsi_ok_now and atr_band_ok_now:
                    # === Step B guard on window-confirmed longs ===
                    if DISABLE_LONG_COMBO and pending["dir"] == "buy":
                        if (ema50 is not None and ema200 is not None) and (ema50 >= ema200) and (rsi <= RSI_LONG_BAD_CUTOFF):
                            direction = None
                            pending = None
                        else:
                            direction = pending["dir"]
                            funnel["rsi_met_in_window"] += 1
                            pending = None
                    else:
                        direction = pending["dir"]
                        funnel["rsi_met_in_window"] += 1
                        pending = None
                

        # Advance MACD state
        prev_macd, prev_signal = macd_line, sig_line

        # If no trade, move to next bar
        if direction is None:
            i += 1
            continue

        # --------- EXECUTION ----------
        if i + 1 >= total_bars:
            break

        # Enter on NEXT bar open
        dt_entry, o1, h1, l1, c1, v1 = data[i + 1]
        entry = o1
        atr_at_entry = atr

        # initial SL/TP from entry ATR
        stop   = entry - atr_at_entry * ATR_MULT_SL if direction == "buy" else entry + atr_at_entry * ATR_MULT_SL
        target = entry + atr_at_entry * ATR_MULT_TP if direction == "buy" else entry - atr_at_entry * ATR_MULT_TP

        # position size
        size = position_size(portfolio, entry, stop)

        # walk forward until exit
        bars_held = 0
        exit_reason = None
        exit_price = None
        j = i + 1

        # for breakeven
        peak = entry
        trough = entry

        while j < total_bars:
            dt_j, oj, hj, lj, cj, vj = data[j]
            bars_held += 1

            # SL/TP checks
            if direction == "buy":
                hit_stop = (lj <= stop)
                hit_tp   = (hj >= target)
                if hit_stop and hit_tp:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_stop:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_tp:
                    exit_price = target; exit_reason = "target"; break
            else:
                hit_stop = (hj >= stop)
                hit_tp   = (lj <= target)
                if hit_stop and hit_tp:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_stop:
                    exit_price = stop; exit_reason = "stop"; break
                elif hit_tp:
                    exit_price = target; exit_reason = "target"; break

            # Optional breakeven nudge after some progress
            if USE_BREAKEVEN:
                if direction == "buy":
                    peak = max(peak, hj)
                    if (peak - entry) >= (BE_MULT * atr_at_entry) and stop < entry:
                        stop = entry
                else:
                    trough = min(trough, lj)
                    if (entry - trough) >= (BE_MULT * atr_at_entry) and stop > entry:
                        stop = entry

            # time exit
            if bars_held >= MAX_BARS_HOLD:
                exit_price = cj
                exit_reason = "time_exit"
                break

            j += 1

        # finalize exit price/time
        if exit_price is None:
            dt_exit, _, _, _, last_c, _ = data[-1]
            exit_price = last_c
            exit_time = dt_exit
            exit_reason = "data_end"
        else:
            exit_time = data[j][0]

        # P&L and equity updates
        if direction == "buy":
            pl = (exit_price - entry) * size
            # R multiple based on initial stop distance
            r_mult = (exit_price - entry) / max((entry - stop), 1e-12)
        else:
            pl = (entry - exit_price) * size
            r_mult = (entry - exit_price) / max((stop - entry), 1e-12)

        portfolio += pl
        peak_equity = max(peak_equity, portfolio)

        # update daily guard AFTER exits too
        daily_dd = (daily_start_equity - portfolio) / daily_start_equity if daily_start_equity > 0 else 0.0
        if daily_dd >= MAX_DAILY_DD:
            daily_blocked = True  # no more entries today

        # track max total DD
        total_dd_now = (peak_equity - portfolio) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, total_dd_now)

        # trade stats
        if pl >= 0:
            n_wins += 1
            gross_profit += pl
        else:
            n_losses += 1
            gross_loss += -pl
        n_trades += 1

        # --- Monthly stats (by EXIT month) ---
        mkey = exit_time.strftime("%Y-%m")
        monthly[mkey]["pl"] += pl
        monthly[mkey]["trades"] += 1
        if pl > 0:
            monthly[mkey]["wins"] += 1
            monthly[mkey]["profits"].append(pl)
        elif pl < 0:
            monthly[mkey]["losses"] += 1
            monthly[mkey]["losses_list"].append(abs(pl))

        # log trade
        if LOG_TRADES:
            write_csv_row(
                TRADES_CSV, trade_fields,
                {
                    "entry_time": dt_entry.isoformat(sep=" "),
                    "exit_time":  exit_time.isoformat(sep=" "),
                    "direction":  direction,
                    "entry":      f"{entry:.5f}",
                    "exit":       f"{exit_price:.5f}",
                    "stop":       f"{stop:.5f}",
                    "target":     f"{target:.5f}",
                    "atr_at_entry": f"{atr_at_entry:.5f}",
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
    print(f" + EMA trend (recent):                {funnel['macd+trend']:,}")
    print(f" + RSI immediate (same bar):          {funnel['macd+trend+rsi_immediate']:,}")
    print(f" + ATR gate immediate:                {funnel['macd+trend+rsi+atr_immediate']:,}")
    print(f" RSI met within +{RSI_WINDOW_AHEAD} bars (entries via window): {funnel['rsi_met_in_window']:,}")
    print(f" RSI window expired (no confirm):     {funnel['rsi_window_expired']:,}")
    for key in ["daily_dd_guard_active","warmup"]:
        if funnel[key]:
            print(f" {key:35} {funnel[key]:,}")

    if examples['ALL']:
        print("\nFirst qualifying bars (all immediate filters passed):")
        for ts, rsi_v, atr_v, e50, e200 in examples["ALL"]:
            print(f"  {ts} | RSI={rsi_v:.1f} ATR={atr_v:.5f} EMA50={e50:.5f} EMA200={e200:.5f}")

    # ======== MONTHLY PERFORMANCE (Jan→Dec) ========
    print("\n======== MONTHLY PERFORMANCE ========")
    if monthly:
        first_key = sorted(monthly.keys())[0]
        year = int(first_key.split("-")[0])
    else:
        year = 2023
    ordered_months = [f"{year}-{m:02d}" for m in range(1, 13)]

    for month in ordered_months:
        m = monthly.get(month, {
            "pl": 0.0, "trades": 0, "wins": 0, "losses": 0,
            "profits": [], "losses_list": []
        })
        profit_sum = sum(m["profits"])
        loss_sum   = sum(m["losses_list"])
        if loss_sum > 1e-12:
            pf_m = profit_sum / loss_sum
        elif profit_sum > 0:
            pf_m = float('inf')
        else:
            pf_m = 0.0
        win_rate = (m["wins"] / m["trades"]) * 100 if m["trades"] else 0.0
        ret_pct  = m["pl"] / ACCOUNT_START
        print(f"{month} | Trades: {m['trades']:3d} | Wins: {m['wins']:3d} | Losses: {m['losses']:3d} "
              f"| Win%: {win_rate:5.1f}% | P/L: ${m['pl']:9.2f} | Return: {ret_pct:6.2%} | PF: {pf_m:4.2f}")

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
