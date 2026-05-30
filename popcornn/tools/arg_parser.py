import argparse
from typing import Optional


def build_default_arg_parser() -> argparse.ArgumentParser:
    """
    Argparse parser used by ``examples/run.py``.

    Single ``--config`` / ``-c`` argument pointing at a YAML config
    file. Kept as a function rather than a hard-coded ``main`` so a
    user script can extend the parser with their own flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file.",
        type=str,
        default="config.yaml"
    )
    return parser