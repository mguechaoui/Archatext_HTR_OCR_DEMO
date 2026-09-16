"""Centralized logging setup. Import `configure_logging()` once at startup."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported under a test runner) — skip.
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers unless we're in debug mode.
    for noisy in ("PIL", "matplotlib", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
