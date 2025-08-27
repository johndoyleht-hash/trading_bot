# tests/test_portfolio_forward.py
import os
import re
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PAIRS = os.environ.get("PORTFOLIO_PAIRS", "EURUSD,GBPUSD,USDJPY")
YEAR  = os.environ.get("PORTFOLIO_YEAR",  "2024")

# Guardrails (your stated targets)
MIN_PF        = float(os.environ.get("PORT_MIN_PF", "1.2"))
MAX_DD        = float(os.environ.get("PORT_MAX_DD", "0.075"))   # 7.5%
MIN_FINAL_EQ  = float(os.environ.get("PORT_MIN_FINAL", "25000"))  # > start eq

@pytest.mark.portfolio
def test_portfolio_forward_health():
    # We expect paper sims to have already run (CI forward job).
    # If any pair is missing, skip (keeps CI green if data not present).
    missing = []
    for pair in PAIRS.split(","):
        p = REPO_ROOT / "runs" / "live_sim" / pair / YEAR
        if not p.exists() or not any(p.glob("trades_*.csv")):
            missing.append(pair)
    if missing:
        pytest.skip(f"Missing live_sim trades for: {', '.join(missing)}")

    # Run portfolio summary (re-uses latest trades)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "portfolio_health.py"),
        "--year", YEAR,
        "--pairs", PAIRS,
        "--use_latest",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"portfolio_health.py failed:\n{proc.stderr}\n{proc.stdout}"

    out = proc.stdout

    # Pull the three metrics from stdout
    pf_m = re.search(r"Profit Factor:\s*([0-9.]+)", out)
    cr_m = re.search(r"Cumulative Return:\s*([-0-9.]+)%", out)
    fe_m = re.search(r"Final Equity:\s*\$([0-9.,]+)", out)
    dd_m = re.search(r"Max Drawdown:\s*([-0-9.]+)%", out)

    assert pf_m and cr_m and fe_m and dd_m, f"Could not parse portfolio output:\n{out}"

    pf = float(pf_m.group(1))
    cum_ret_pct = float(cr_m.group(1))
    final_eq = float(fe_m.group(1).replace(",", ""))
    max_dd_pct = float(dd_m.group(1))

    # Assertions
    assert pf >= MIN_PF, f"Portfolio PF {pf:.2f} < {MIN_PF}"
    assert (max_dd_pct / 100.0) <= MAX_DD, f"Max DD {max_dd_pct:.2f}% > {MAX_DD*100:.2f}%"
    assert final_eq >= MIN_FINAL_EQ, f"Final equity ${final_eq:,.2f} < ${MIN_FINAL_EQ:,.0f}"

    # Nice-to-haves (won't fail CI)
    print(f"[portfolio_ok] PF={pf:.2f}  MaxDD={max_dd_pct:.2f}%  FinalEq=${final_eq:,.2f}  CumRet={cum_ret_pct:.2f}%")
