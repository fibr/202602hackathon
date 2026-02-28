"""Centralized logging setup. Logs to console + timestamped file in logs/.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""

import logging
import os
import sys
from datetime import datetime

_initialized = False
_log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
_log_file = None


def get_log_file():
    """Return the current log file path."""
    return _log_file


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger that writes to console (INFO) and file (DEBUG).

    All loggers share the same file handler so the full session
    ends up in one timestamped log file.
    """
    global _initialized, _log_file

    if not _initialized:
        _initialized = True
        os.makedirs(_log_dir, exist_ok=True)
        _log_file = os.path.join(
            _log_dir,
            datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        )

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # File handler: DEBUG level, with timestamps
        fh = logging.FileHandler(_log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-5s [%(name)s] %(message)s',
            datefmt='%H:%M:%S.%f'
        ))
        root.addHandler(fh)

        # Console handler: INFO level, minimal format
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(ch)

    return logging.getLogger(name or 'app')
