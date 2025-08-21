# tests/test_core_smoke.py
import csv
import os
from datetime import datetime
from src import baseline_core as baseline

def _slice_csv(src_path: str, dst_path: str, max_rows: int = 5000):
    """Copy header + first max_rows rows to dst_path."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", newline="") as fin, open(dst_path, "w", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        for i, row in enumerate(r):
            w.writerow(row)
            if i >= max_rows:  # i=0 is header
                break

def test_core_smoke_small_csv(tmp_path):
    # 1) Point to your real 2023 EURUSD file (baseline has default)
    src_csv = baseline.DATA_CSV_PATH
    assert os.path.exists(src_csv), f"Missing source CSV at {src_csv}"

    # 2) Make a tiny slice (fast)
    dst_csv = tmp_path / "EURUSD_2023_slice.csv"
    _slice_csv(src_csv, str(dst_csv), max_rows=5000)

    # 3) Run backtest on the slice
    res = baseline.run_backtest(DATA_CSV_PATH=str(dst_csv), PAIR="EURUSD")

    # 4) Basic shape checks
    assert isinstance(res, dict)
    for key in ["monthly_results", "account_start", "funnel", "examples", "summary"]:
        assert key in res, f"Missing key in result: {key}"

    # 5) Summary sanity
    s = res["summary"]
    for k in ["trades", "wins", "losses", "wl_ratio", "pf", "expectancy", "cum_ret", "final_equity", "max_dd"]:
        assert k in s, f"Missing summary field: {k}"

    # 6) Should have processed *some* trades on the slice
    assert s["trades"] >= 0  # not guaranteeing >0 on a small slice, just that it ran
# tests/test_full_year_2023.py
import pytest
from src import baseline_core as baseline

@pytest.mark.slow
def test_full_year_2023_runs():
    res = baseline.run_backtest(DATA_CSV_PATH=baseline.DATA_CSV_PATH, PAIR="EURUSD")
    assert res["summary"]["trades"] > 0
