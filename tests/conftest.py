"""Shared pytest configuration for popcornn's tests.

Holds what every test module relies on, so it is defined once:

- test ids that name data files by file name (``pytest_make_parametrize_id``).
"""

from __future__ import annotations

from pathlib import Path

from tests import IMAGES_DIR


def pytest_make_parametrize_id(config, val, argname) -> str | None:
    """Name a test-data path by its file name in test ids.

    Paths are absolute (``IMAGES_DIR``), so without this the ids would contain
    the checkout's location and differ between machines.
    """
    if isinstance(val, str) and val.startswith(str(IMAGES_DIR)):
        return Path(val).name
    return None
