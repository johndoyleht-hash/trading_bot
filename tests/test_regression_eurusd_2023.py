# tests/test_regression_eurusd_2023.py
import pytest
pytestmark = pytest.mark.regression
# ... rest of file ...

# tests/test_regression_eurusd_2023.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import baseline_core as baseline

def _abs_close(a, b, tol):
    return abs(a - b) <= tol

def test_eurusd_2023_regression():
    data_path = os.path.join(
        ROOT, "data", "clean", "EURUSD", "2023", "EURUSD_1y_2023_clean.csv"
    )
    # If data isn’t present in CI, skip instead of failing
    if not os.path.exists(data_path):
        import pytest
        pytest.skip(f"Missing data file: {data_path}")

    res = baseline.run_backtest(DATA_CSV_PATH=data_path, PAIR="EURUSD")
    s = res["summary"]

    # locked from your matching run
    EXP_TRADES = 288
    EXP_PF = 1.38
    EXP_CUM_RET = 0.2084     # 20.84%
    EXP_FINAL_EQ = 30211.04
    EXP_MAX_DD = 0.0424      # 4.24%

    # tolerances (very tight, but avoids float jitter)
    assert s["trades"] == EXP_TRADES
    assert _abs_close(s["pf"], EXP_PF, 0.01)
    assert _abs_close(s["cum_ret"], EXP_CUM_RET, 0.0005)
    assert _abs_close(s["final_equity"], EXP_FINAL_EQ, 3.0)
    assert _abs_close(s["max_dd"], EXP_MAX_DD, 0.0005)
