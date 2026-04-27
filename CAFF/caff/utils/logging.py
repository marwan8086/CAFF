"""
caff/utils/logging.py
=====================
Centralized logging setup. All modules use Python's standard
`logging` module — never `print` for diagnostic output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """Configure root logger to stream to stdout and (optionally) a file.

    Parameters
    ----------
    level : str
        One of DEBUG, INFO, WARNING, ERROR.
    log_file : path or None
        If set, ALSO write logs to this file.
    fmt : str
        Format string for log records.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    formatter = logging.Formatter(fmt)
    for h in handlers:
        h.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(getattr(logging, level.upper()))

    # Quiet down noisy third-party loggers
    for noisy in ("urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)