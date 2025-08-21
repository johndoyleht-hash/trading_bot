# fast_backtest_gold.py
# Full-year backtest (EURUSD M1 2024), streaming indicators, next-bar entries,
# SL/TP/time exits, DAILY & TOTAL DD guards, RSI window confirm, ATR percentile band,
# EMA "recent" trend, breakeven move, optional trailing stop, session filter,
# basic structure gates (MACD separation, EMA gap), momentum/body filter,
# and MONTHLY performance breakdown.

import csv
from datetime import datetime
import os
from collections import Counter, deque, defaultdict
import bisect

# ========= Speed / Logging =========
FAST_MODE      = True
LOG_TRADES     = True
PROGRESS_EVERY = 50_000

# ========= Data / Strategy Config =========
DATA_CSV_PATH   = "EURUSD_1y_2024_clean.csv"  # timestamp,open,high,low,close,volume
ACCOUNT_START   = 25_000.00
RISK_PER_TRADE  = 0.005        # 0.5% per trade
MACD_EPS        = 1e-6
ATR_MULT_SL     = 1.5          # 1.5R stop
ATR_MULT_TP     = 3.0          # 3R target
MAX_BARS_HOLD   = 300          # ~5 hours on M1

# ======== Entry filter thresholds (balanced) ========
RSI_BUY_MAX     = 45.0         # buy needs RSI <= 35
RSI_SELL_MIN    = 55.0         # sell needs RSI >= 65
RSI_WINDOW_AHEAD = 6           # RSI may confirm within +6 bars after cross

# ======== ATR gate via rolling percentile band ========
ATR_BAND_LOOKBACK = 1000
ATR_P_LOW         = 0.10
ATR_P_HIGH        = 0.98
ATR_MIN_WARMUP    = 150        # permissive until we have this many samples

# ======== EMA "recent" trend ========
EMA_TREND_RECENT  = 8          # trend may be true on any of last N bars

# ======== Structure / quality gates ========
MACD_MIN_SEP      = 6.0e-6     # min |MACD - signal| at cross
MIN_EMA_GAP_ATR   = 0.18       # |EMA50-EMA200| >= 0.22 * ATR

# ======== Momentum / body requirement ========
BODY_ATR_MULT     = 0.15       # confirm bar needs body >= 0.20 * ATR
MOMENTUM_BODY_ATR = 0.20       # momentum override: if body >= this, ignore ATR band

# ======== Breakeven & Trailing ========
BE_MULT           = 0.9        # move stop to BE after 0.8 * ATR favorable
TRAIL_ARM_R       = 1.5        # arm ATR trail once unrealized >= 1.5R
ATR_TRAIL_MULT    = 1.0        # trail distance = 1.0 * entry ATR

# ======== Session filter (disable for 24/5 to get more trades) ========
USE_SESSION_FILTER   = False
SESSION_START_HHMM   = 600     # 06:00 UTC
SESSION_END_HHMM     = 2100    # 21:00 UTC
TRADE_WEEKDAYS_ONLY  = True

# ========= Risk Guards =========
MAX_DAILY_DD          = 0.04   # 4% from start-of-day equity
MAX_TOTAL_DD          = 0.07   # 7% from peak equity
MAX_DAILY_LOSS_STREAK = 3      # throttle after N daily losses
DAILY_PEAK_DD_LIMIT   = 0.03   # 3% from today's peak equity

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
        # Permissive during warmup so we don't starve early trades
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
    rsi_prev = None
    prev_macd = None
    prev_signal = None

    portfolio = ACCOUNT_START
    peak_equity = ACCOUNT_START

    # Daily guard state
    current_day = None
    daily_start_equity = ACCOUNT_START
    daily_blocked = False
    daily_loss_streak = 0
    daily_peak_equity = ACCOUNT_START

    # Perf aggregates
    n_trades = n_wins = n_losses = 0
    gross_profit = 0.0
    gross_loss   = 0.0
    max_drawdown = 0.0

    # Signal funnel counters
    funnel = Counter()
    examples = {"ALL": []}

    # ======== MONTHLY PERFORMANCE (Jan→Dec) ========
print("\n======== MONTHLY PERFORMANCE ========")
if monthly:
    first_key = sorted(monthly.keys())[0]
    year = int(first_key.split("-")[0])
else:
    year = 2024
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
    print(
        f"{month} | Trades: {m['trades']:3d} | Wins: {m['wins']:3d} | Losses: {m['losses']:3d} "
        f"| Win%: {win_rate:5.1f}% | P/L: ${m['pl']:9.2f} | Return: {ret_pct:6.2%} | PF: {pf_m:4.2f}"
    )

    # Pending signal after MACD cross awaiting RSI confirmation
    pending = None  # {'dir': 'buy'/'sell', 'expire': int, 'cross_idx': int}

    # Histories for recent checks and ATR band
    ema50_hist, ema200_hist = [], []
    atr_rp = RollingPercentile(ATR_BAND_LOOKBACK)

    # Monthly accumulator: YYYY-MM → stats
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

        # progress ping
        if FAST_MODE and (i % PROGRESS_EVERY == 0) and i > 0:
            pct = (i / total_bars) * 100
            print(f"Processed {i:,}/{total_bars:,} bars ({pct:.2f}%)")

        # Day rollover
        if current_day is None:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False
            daily_loss_streak = 0
            daily_peak_equity = portfolio
        elif dt.date() != current_day:
            current_day = dt.date()
            daily_start_equity = portfolio
            daily_blocked = False
            daily_loss_streak = 0
            daily_peak_equity = portfolio

        # Update indicators
        vals = ind.update(h, l, c)
        ema50, ema200 = vals["ema50"], vals["ema200"]
        rsi, atr = vals["rsi"], vals["atr"]
        macd_line, sig_line = vals["macd"], vals["signal"]
                # --- RSI momentum (slope) flags ---
        # For buys, "rising" RSI is acceptable momentum; for sells, "falling" RSI is acceptable
        rsi_momo_up   = (rsi_prev is not None and rsi is not None and rsi > rsi_prev)
        rsi_momo_down = (rsi_prev is not None and rsi is not None and rsi < rsi_prev)
        
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

        # Enforce drawdown guards
        total_dd = (peak_equity - portfolio) / peak_equity if peak_equity > 0 else 0.0
        daily_dd = (daily_start_equity - portfolio) / daily_start_equity if daily_start_equity > 0 else 0.0
        daily_peak_equity = max(daily_peak_equity, portfolio)
        daily_peak_dd = (daily_peak_equity - portfolio) / daily_peak_equity if daily_peak_equity > 0 else 0.0

        if total_dd >= MAX_TOTAL_DD:
            print("⛔ Total DD limit reached — stopping backtest.")
            break
        if (daily_blocked or daily_dd >= MAX_DAILY_DD or
            daily_loss_streak >= MAX_DAILY_LOSS_STREAK or daily_peak_dd >= DAILY_PEAK_DD_LIMIT):
            daily_blocked = True
            funnel["daily_dd_guard_active"] += 1
            prev_macd, prev_signal = macd_line, sig_line
            i += 1
            continue

        # Strict MACD cross on this bar
        buy_cross  = (prev_macd <= prev_signal) and ((macd_line - sig_line) >  MACD_EPS)
        sell_cross = (prev_macd >= prev_signal) and ((macd_line - sig_line) < -MACD_EPS)

        # Pre-compute quality gates on this bar
        macd_sep_ok = (macd_line is not None and sig_line is not None and abs(macd_line - sig_line) >= MACD_MIN_SEP)
        ema_gap_ok  = (ema50 is not None and ema200 is not None and atr is not None and
                       abs(ema50 - ema200) >= (MIN_EMA_GAP_ATR * atr))

        # ----- Funnel accounting (immediate; with body & momentum override) -----
        direction = None
        if buy_cross or sell_cross:
            funnel["macd_cross"] += 1
            dir_now = "buy" if buy_cross else "sell"
            trend_ok_now = trend_recent_ok(ema50_hist, ema200_hist, EMA_TREND_RECENT, dir_now)
            if trend_ok_now:                 funnel["macd+trend"] += 1
            if macd_sep_ok:                  pass
            if ema_gap_ok:                   pass
            pending = {"dir": dir_now, "expire": i + RSI_WINDOW_AHEAD, "cross_idx": i}
            if dir_now == "buy":
                rsi_immediate_ok = (rsi < RSI_BUY_MAX) or rsi_momo_up
            else:
                rsi_immediate_ok = (rsi > RSI_SELL_MIN) or rsi_momo_down
            atr_band_ok_now = atr_rp.band_ok(atr, ATR_P_LOW, ATR_P_HIGH)
            momentum_override = (abs(c - o) >= MOMENTUM_BODY_ATR * (atr or 0.0))
            atr_ok_final = atr_band_ok_now or momentum_override
            
            # NOTE: body is NOT required for immediate entry anymore
            if trend_ok_now and macd_sep_ok and ema_gap_ok and rsi_immediate_ok and atr_ok_final:
                if len(examples["ALL"]) < 5:
                    examples["ALL"].append((dt.isoformat(sep=" "), float(rsi), float(atr), float(ema50), float(ema200)))
                direction = dir_now
                pending = None
            # Otherwise start/refresh pending (we'll demand body on the confirm bar)
            elif trend_ok_now and macd_sep_ok and ema_gap_ok:
                pending = {"dir": dir_now, "expire": i + RSI_WINDOW_AHEAD, "cross_idx": i}

        # ----- Pending confirm within window -----
        if direction is None and pending is not None:
            if i > pending["expire"]:
                funnel["rsi_window_expired"] += 1
                pending = None
            else:
                want_buy = (pending["dir"] == "buy")
                trend_ok_recent = trend_recent_ok(ema50_hist, ema200_hist, EMA_TREND_RECENT, pending["dir"])
            if want_buy:
                rsi_ok_now = (rsi < RSI_BUY_MAX) or rsi_momo_up
            else:
                rsi_ok_now = (rsi > RSI_SELL_MIN) or rsi_momo_down
                ema_gap_ok_confirm = (ema50 is not None and ema200 is not None and atr is not None and
                                      abs(ema50 - ema200) >= (MIN_EMA_GAP_ATR * atr))
                atr_band_ok_now = atr_rp.band_ok(atr, ATR_P_LOW, ATR_P_HIGH)
                body_ok_now = (abs(c - o) >= BODY_ATR_MULT * (atr or 0.0))
                momentum_override = (abs(c - o) >= MOMENTUM_BODY_ATR * (atr or 0.0))
                atr_ok_final = atr_band_ok_now or momentum_override
                
                if trend_ok_recent and ema_gap_ok_confirm and rsi_ok_now and body_ok_now and atr_ok_final:
                    direction = pending["dir"]
                    funnel["rsi_met_in_window"] += 1
                    pending = None
                
        # Advance MACD state for next bar decision
        prev_macd, prev_signal = macd_line, sig_line

        # If no trade, move on
        if direction is None:
            rsi_prev = rsi
            i += 1
            continue

        # --------- EXECUTION ----------
        if i + 1 >= total_bars:
            break

        # Enter on next bar open
        dt_entry, o1, h1, l1, c1, v1 = data[i + 1]
        entry = o1
        atr_at_entry = atr if atr is not None else 0.0

        # initial SL/TP from entry ATR
        stop   = entry - atr_at_entry * ATR_MULT_SL if direction == "buy" else entry + atr_at_entry * ATR_MULT_SL
        target = entry + atr_at_entry * ATR_MULT_TP if direction == "buy" else entry - atr_at_entry * ATR_MULT_TP

        # position size and risk per trade (distance to initial stop in price)
        size = position_size(portfolio, entry, stop)
        risk_at_entry = max(atr_at_entry * ATR_MULT_SL, 1e-12)  # used for R-multiple, trail arming

        # walk forward until exit
        bars_held = 0
        exit_reason = None
        exit_price = None
        j = i + 1

        # excursions for BE and trailing
        peak = entry
        trough = entry
        trail_active = False
        trail_stop = None

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

            # --- Breakeven move ---
            if direction == "buy":
                peak = max(peak, hj)
                if (peak - entry) >= (BE_MULT * atr_at_entry) and stop < entry:
                    stop = entry
            else:
                trough = min(trough, lj)
                if (entry - trough) >= (BE_MULT * atr_at_entry) and stop > entry:
                    stop = entry

            # --- Trailing stop (armed after TRAIL_ARM_R) ---
            if not trail_active:
                if direction == "buy":
                    unrealized_r = (hj - entry) / risk_at_entry
                else:
                    unrealized_r = (entry - lj) / risk_at_entry
                if unrealized_r >= TRAIL_ARM_R:
                    trail_active = True
                    if direction == "buy":
                        trail_stop = max(stop, cj - ATR_TRAIL_MULT * atr_at_entry)
                    else:
                        trail_stop = min(stop, cj + ATR_TRAIL_MULT * atr_at_entry)
                    stop = trail_stop
            else:
                # ratchet trail; never loosen
                if direction == "buy":
                    new_trail = cj - ATR_TRAIL_MULT * atr_at_entry
                    if new_trail > (trail_stop if trail_stop is not None else -1e9):
                        trail_stop = new_trail
                        stop = max(stop, trail_stop)
                else:
                    new_trail = cj + ATR_TRAIL_MULT * atr_at_entry
                    if new_trail < (trail_stop if trail_stop is not None else 1e9):
                        trail_stop = new_trail
                        stop = min(stop, trail_stop)

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
            r_mult = (exit_price - entry) / risk_at_entry
        else:
            pl = (entry - exit_price) * size
            r_mult = (entry - exit_price) / risk_at_entry

        portfolio += pl
        peak_equity = max(peak_equity, portfolio)

        # update daily guard AFTER exits
        daily_dd = (daily_start_equity - portfolio) / daily_start_equity if daily_start_equity > 0 else 0.0
        if daily_dd >= MAX_DAILY_DD:
            daily_blocked = True

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

        # loss streak
        daily_loss_streak = (daily_loss_streak + 1) if pl < 0 else 0

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

# print("\n======== MONTHLY PERFORMANCE ========")
print("\n======== MONTHLY PERFORMANCE ========")
if monthly:
    first_key = sorted(monthly.keys())[0]
    year = int(first_key.split("-")[0])
else:
    year = 2024

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
