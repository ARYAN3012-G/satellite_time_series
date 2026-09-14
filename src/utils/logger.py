"""
utils/logger.py -- shared logging setup used by every module.
One consistent format, writing to both console and logs/pipeline.log,
so issues surfaced in Module 2+ (duplicate dates, gaps, scaling checks,
etc.) are never lost -- they're in a persistent log file, not just
scrolled past in a notebook cell.
"""
import logging
import sys
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "pipeline.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=logging.INFO,
            format=_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(_LOG_FILE, encoding="utf-8"),
            ],
        )
        _configured = True
    return logging.getLogger(name)
