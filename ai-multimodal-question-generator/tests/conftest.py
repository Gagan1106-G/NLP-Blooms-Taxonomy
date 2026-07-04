"""Pytest configuration and shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable from all test modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
