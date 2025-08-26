![tests](https://github.com/johndoyleht-hash/trading_bot/actions/workflows/test.yml/badge.svg)

## Live/Forward Runbook

**Pairs in scope**: EURUSD (primary), GBPUSD, USDJPY  
**Forward configs**:
- `configs/forward/gbpusd_2024_forward.yaml`
- `configs/forward/usdjpy_2024_forward.yaml`
- (EURUSD uses baseline_2024_EURUSD.yaml or your forward variant)

**Risk knobs (per pair)**:
- RSI bands: documented in each YAML
- ATR floor: pair-tuned; low ATR is filtered out
- Global DD guard (equity): 7% stop for forward tests
- Per-trade sizing: as in YAML (fractional of equity)

**Expected ranges (per pair)**:
- EURUSD (2024): PF ~1.55, DD ~4–5%, trades ~30–40/mo
- GBPUSD (2024 tuned): PF ~1.25–1.36, DD ~4–5%, trades ~12–18/mo
- USDJPY (2024 tuned): PF ~1.16–1.25+, DD ~4–5%, trades ~18–30/mo

**How to run locally**
- Quick checks: `make smoke`, `make regression`, `make forward`
- One pair summary: `make summary pair=EURUSD year=2024`
- Paper/live-sim (see scripts below): `make live pair=EURUSD cfg=configs/forward/...`
