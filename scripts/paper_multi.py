# scripts/paper_multi.py
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--pairs", required=True, help="Comma-separated pairs, e.g. EURUSD,GBPUSD,USDJPY")
    parser.add_argument("--cfg_dir", required=True, help="Directory containing forward YAML configs")
    args = parser.parse_args()

    year = args.year
    pairs = args.pairs.split(",")
    cfg_dir = Path(args.cfg_dir)

    for pair in pairs:
        cfg_path = cfg_dir / f"{pair.lower()}_{year}_forward.yaml"
        if not cfg_path.exists():
            print(f"⚠️ Config missing: {cfg_path}")
            continue

        print(f"\n=== Running paper sim for {pair} {year} ===")
        subprocess.run(
            [
                "python3", "scripts/paper_sim.py",
                "--pair", pair,
                "--year", str(year),
                "--config", str(cfg_path),
            ],
            check=True
        )

if __name__ == "__main__":
    main()
