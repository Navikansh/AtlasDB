"""
Logging Setup
---------------
Call `configure_logging()` once at process startup (API app startup, CLI entry points, benchmark scripts). 
Every module below gets its logger via `logging.getLogger("atlasdb.<module>")` and never calls print() directly.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger("atlasdb")
    root.setLevel(level.upper())

    if root.handlers:
        return  # already configured, don't double-attach handlers

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
