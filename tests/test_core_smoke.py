# tests/test_core_smoke.py
import csv
import os
import sys
from datetime import datetime, timedelta
import importlib  # (not strictly needed now, but fine to keep)

# Make repo root importable: add parent of this tests/ dir to sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _write_synthetic_csv(path: str, n_rows: int = 1200):
    """
    Create a tiny, self-contained M1-like OHLCV file so CI doesn't need your big data folder.
    Price is a gentle random walk that stays positive.
    """
    start = datetime(2023, 1, 3, 0, 0, 0)
    price = 1.07
    import random
    random.seed(42)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","open","high","low","close","volume"])
        t = start
        for _ in range(n_rows):
            # small drift + noise
            delta = (random.random() - 0.5) * 0.0004
            o = price
            h = o + abs(delta) * 1.5
            l = o - abs(delta) * 1.5
            c = o + delta
            price = max(0.5, c)  # keep positive-ish
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), f"{o:.5f}", f"{h:.5f}", f"{l:.5f}", f"{c:.5f}", "0"])
            t += timedelta(minutes=1)

def test_backtest_runs_on_synthetic_data(tmp_path, baseline_module):
    # Make a tiny fake EURUSD 2023 path
    csv_path = tmp_path / "data" / "clean" / "EURUSD" / "2023" / "EURUSD_1y_2023_clean.csv"
    _write_synthetic_csv(str(csv_path), n_rows=1200)  # ~20 hours of M1 bars

    # Ensure a clean config in baseline_core
    # (override a few heavy settings to keep it fast)
    baseline_module.DATA_CSV_PATH = str(csv_path)
    baseline_module.ACCOUNT_START = 25_000.0
    baseline_module.RISK_PER_TRADE = 0.005
    baseline_module.FAST_MODE = True
    baseline_module.LOG_TRADES = False
    baseline_module.PROGRESS_EVERY = 10**9  # effectively disables progress prints
    baseline_module.USE_SESSION_FILTER = False  # keep all rows in synthetic
    baseline_module.EXCLUDE_UTC_HOURS = set()
    baseline_module.EXCLUDE_WEEKDAYS = set()
    baseline_module.MAX_BARS_HOLD = 120  # shorten exits for speed
    baseline_module.USE_APRIL_ATR_TIGHTENING = False

    # Run
    res = baseline_module.run_backtest(DATA_CSV_PATH=str(csv_path), PAIR="EURUSD")
    assert isinstance(res, dict)
    assert "summary" in res and "monthly_results" in res

    s = res["summary"]
    # Just sanity checks — we don’t care about exact PnL on synthetic data
    assert s["trades"] >= 1
    assert 0 <= s["max_dd"] <= 1
