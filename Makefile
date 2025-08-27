# ===== Makefile =====
# Usage examples:
#   make                 # show help
#   make smoke           # run non-regression, non-forward tests
#   make regression      # run locked baseline tests
#   make forward         # run forward-health tests
#   make ci-local        # run what CI does: smoke -> regression -> forward
#   make run pair=GBPUSD year=2024 cfg=configs/forward/gbpusd_2024_forward.yaml
#   make summary pair=GBPUSD year=2024
#   make lfs-verify      # sanity check that clean CSVs are tracked via LFS

SHELL := /bin/bash
PY := python3

# Default "help" target
.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  make smoke         - run quick tests (not regression/forward)"
	@echo "  make regression    - run locked regression matrix"
	@echo "  make forward       - run forward health checks"
	@echo "  make ci-local      - run smoke -> regression -> forward"
	@echo "  make run pair=PA year=YYYY cfg=path/to.yaml"
	@echo "                     - run baseline_pair_runner with a config"
	@echo "  make summary pair=PA year=YYYY"
	@echo "                     - print one-line JSON summary for pair/year"
	@echo "  make lfs-verify    - verify clean CSVs are tracked via Git LFS"
	@echo ""

# -------- Pytest wrappers --------
.PHONY: smoke regression forward ci-local
smoke:
	$(PY) -m pytest -q -m "not regression and not forward"

regression:
	$(PY) -m pytest -q -m regression

forward:
	$(PY) -m pytest -q -m forward

ci-local: smoke regression forward

# -------- Ad-hoc baseline run & summary --------
# Ex: make run pair=GBPUSD year=2024 cfg=configs/forward/gbpusd_2024_forward.yaml
.PHONY: run
run:
	@if [ -z "$(pair)" ] || [ -z "$(year)" ] || [ -z "$(cfg)" ]; then \
	  echo "Usage: make run pair=PAIR year=YYYY cfg=CONFIG.yaml"; exit 2; \
	fi
	$(PY) scripts/baseline_pair_runner.py --pair $(pair) --year $(year) --config $(cfg)

# Ex: make summary pair=GBPUSD year=2024
.PHONY: summary
summary:
	@if [ -z "$(pair)" ] || [ -z "$(year)" ]; then \
	  echo "Usage: make summary pair=PAIR year=YYYY"; exit 2; \
	fi
	$(PY) scripts/print_summary.py --pair $(pair) --year $(year)

# -------- Git LFS sanity check --------
.PHONY: lfs-verify
lfs-verify:
	@echo "Listing files under data/clean tracked by Git LFS (pointers):"
	@git lfs ls-files | sed -n '1,200p'
	@echo ""
	@echo "Verifying pointer headers for first few CSVs:"
	@/bin/bash -c 'for f in $$(find data/clean -type f -name "*_clean.csv" | head -n 3); do \
	  echo "== $$f =="; \
	  git cat-file -p :$$f 2>/dev/null | sed -n "1,3p" || echo "(not staged)"; \
	done'
paper_multi:
	python3 scripts/paper_multi.py --year $(year) --pairs $(pairs) --cfg_dir configs/forward
portfolio:
	python3 scripts/portfolio_health.py --year $(year) --pairs $(pairs) --use_latest
