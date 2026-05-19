"""
Structured logging setup for the Legal RAG Assistant.

Call `setup_logging()` once at application startup. All modules then
use the standard `logging.getLogger(__name__)` pattern.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.settings import get_settings


def setup_logging() -> None:
    """Configure root logger with console + file handlers."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Formatter ─────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)

    # ── File handler ──────────────────────────────────────────────────
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    # ── Root logger ───────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialized — level=%s, file=%s", settings.log_level, log_path
    )
