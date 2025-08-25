# tests/test_forward_health.py
import os, sys, importlib
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import baseline_core as baseline

@pytest.mark.forward
@pytest.mark.parametrize("pair,year", [
    ("EURUSD", 2024),
    ("GBPUSD", 2024),
    ("USDJPY", 2024),
])
def test_forward_health(pair, year):
    p_local = REPO_ROOT / "data" / "clean" / pair / str(year) / f"{pair}_1y_{year}_clean.csv"
    p_alt   = REPO_ROOT / f"{pair}_1y_{year}_clean.csv"
    p = p_local if p_local.exists() else p_alt
    if not p.exists():
        pytest.skip(f"no data for {pair} {year}: {p}")

    importlib.reload(baseline)
    res = baseline.run_backtest(DATA_CSV_PATH=str(p), PAIR=pair)
    s = res["summary"]

    # loose, “don’t be broken” gates
    assert s["trades"] >= 50
    assert -0.30 <= s["cum_ret"] <= 1.00
    assert 0.00  <= s["max_dd"]  <= 0.25
