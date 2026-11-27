"""popcornn test suite: shared constants for the test modules."""

from __future__ import annotations

from pathlib import Path

# Test data, found relative to this package so tests run from any directory.
IMAGES_DIR = Path(__file__).parent / "images"

# Seed of torch's global generator at the start of every test (conftest.py).
TEST_SEED = 0
