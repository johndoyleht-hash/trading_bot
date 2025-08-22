import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import baseline_core as baseline
import pytest

def _abs_close(a, b, tol): return abs(a - b) <= tol

@pytest.mark.slow
def test_eurusd_2024_regression():
    data_path = os.path.join(
        ROOT, "data", "clean", "EURUSD", "2024", "EURUSD_1y_2024_clean.csv"
    )
    if not os.path.exists(data_path):
        pytest.skip(f"Missing data file: {data_path}")

    res = baseline.run_backtest(DATA_CSV_PATH=data_path, PAIR="EURUSD")
    s = res["summary"]

    # put YOUR current 2024 baseline numbers here after one run from the runner
    EXP_TRADES = 382
    EXP_PF = 1.55
    EXP_CUM_RET = 0.3955
    EXP_FINAL_EQ = 34886.61
    EXP_MAX_DD = 0.0441

    assert s["trades"] == EXP_TRADES
    assert _abs_close(s["pf"], EXP_PF, 0.01)
    assert _abs_close(s["cum_ret"], EXP_CUM_RET, 0.001)
    assert _abs_close(s["final_equity"], EXP_FINAL_EQ, 5.0)
    assert _abs_close(s["max_dd"], EXP_MAX_DD, 0.001)
