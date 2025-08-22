# tests/conftest.py
import os
import sys
import importlib
import pytest

# Make repo root importable (prepend so it wins over site-packages)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

@pytest.fixture()
def baseline_module():
    """
    Import a fresh src.baseline_core for each test to avoid module-level
    globals leaking between tests.
    """
    # Remove any cached import first
    if "src.baseline_core" in sys.modules:
        del sys.modules["src.baseline_core"]

    # Fresh import
    mod = importlib.import_module("src.baseline_core")

    # (Optional) reload immediately to be extra sure
    importlib.reload(mod)
    return mod
