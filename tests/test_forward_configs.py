# tests/test_forward_configs.py
import os, sys, re, json, subprocess, shlex
from pathlib import Path
import pytest

# Resolve repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def discover_forward_cases():
    """
    Find forward YAMLs and infer (PAIR, YEAR) from filename.
    Expected filename patterns:
      - gbpusd_2024_forward.yaml
      - ..._2024_GBPUSD.yaml
      - ... (we try several regexes)
    """
    cases = []
    cfg_dir = REPO_ROOT / "configs" / "forward"
    if not cfg_dir.exists():
        return cases

    for cfg in sorted(cfg_dir.glob("*.yaml")):
        name = cfg.name

        # Try common patterns like gbpusd_2024_forward.yaml
        m = re.search(r"([a-z]{6})[_-](20\d{2})", name, re.IGNORECASE)
        if m:
            pair = m.group(1).upper()
            year = int(m.group(2))
        else:
            # Try baseline_2024_GBPUSD.yaml or FOO_..._GBPUSD_2024.yaml
            m2 = re.search(r"(20\d{2}).*?([A-Za-z]{6})", name)
            m3 = re.search(r"([A-Za-z]{6}).*?(20\d{2})", name)
            if m2:
                year = int(m2.group(1))
                pair = m2.group(2).upper()
            elif m3:
                pair = m3.group(1).upper()
                year = int(m3.group(2))
            else:
                # Could not infer; skip cleanly
                continue

        if not re.fullmatch(r"[A-Z]{6}", pair):
            continue

        csv_path = REPO_ROOT / "data" / "clean" / pair / str(year) / f"{pair}_1y_{year}_clean.csv"
        cases.append((pair, year, cfg, csv_path))

    return cases

CASES = discover_forward_cases()

@pytest.mark.forward
@pytest.mark.parametrize("pair,year,cfg,csv_path", CASES)
def test_forward_health(pair, year, cfg, csv_path):
    # Skip gracefully if data file is missing (keeps workflow green if a pair/year isn’t present)
    if not csv_path.exists():
        pytest.skip(f"Missing data file for {pair} {year}: {csv_path}")

    py = sys.executable  # use current venv's python

    # 1) Run the pair runner with the forward YAML
    cmd_runner = [
        py, str(REPO_ROOT / "scripts" / "baseline_pair_runner.py"),
        "--pair", pair,
        "--year", str(year),
        "--config", str(cfg),
    ]
    # We allow non-zero exit if the runner stops early on DD guard; the summary step still runs.
    subprocess.run(cmd_runner, cwd=str(REPO_ROOT), check=False)

    # 2) Get a clean JSON summary using your print_summary.py
    cmd_summary = [
        py, str(REPO_ROOT / "scripts" / "print_summary.py"),
        "--pair", pair,
        "--year", str(year),
    ]
    out = subprocess.check_output(cmd_summary, cwd=str(REPO_ROOT), text=True)
    # If multiple JSON lines were printed, take the last JSON-looking line
    line = [ln for ln in out.strip().splitlines() if ln.strip().startswith("{")][-1]
    s = json.loads(line)

    # --- Loose guardrails (health, not regression) ---
    assert s["trades"] >= 80, f"{pair} {year}: too few trades: {s['trades']}"
    assert s["pf"] >= 1.05, f"{pair} {year}: low PF: {s['pf']:.2f}"
    assert 0.0 <= s["max_dd"] <= 0.10, f"{pair} {year}: DD too high: {s['max_dd']:.3f}"
    assert -0.20 <= s["cum_ret"] <= 1.20, f"{pair} {year}: cum_ret out of bounds: {s['cum_ret']:.3f}"
