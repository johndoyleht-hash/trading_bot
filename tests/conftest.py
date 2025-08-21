# tests/conftest.py
import os, sys
# Add repo root to sys.path so "from src import baseline_core" works in tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
