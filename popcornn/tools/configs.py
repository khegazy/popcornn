import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from ase.io import read, write


def import_yaml(address):
    """Load a YAML file. Thin ``yaml.safe_load`` wrapper."""
    with open(address, 'r') as file:
        loaded_yaml = yaml.safe_load(file)
    return loaded_yaml


def import_run_config(name):
    """
    Load the run config used by ``examples/run.py``.

    Currently equivalent to ``import_yaml``; kept as a separate
    function so future config validation can hook in here without
    touching every call site.
    """
    yaml_config = import_yaml(name)
    return yaml_config
