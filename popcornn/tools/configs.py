from __future__ import annotations

from pathlib import Path

import yaml


def import_yaml(address):
    """Load a YAML file. Thin ``yaml.safe_load`` wrapper."""
    with Path(address).open() as file:
        return yaml.safe_load(file)


def import_run_config(name):
    """
    Load the run config used by ``examples/run.py``.

    Currently equivalent to ``import_yaml``; kept as a separate
    function so future config validation can hook in here without
    touching every call site.
    """
    return import_yaml(name)
