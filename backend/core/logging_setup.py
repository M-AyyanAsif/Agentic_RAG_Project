"""Centralized logging setup with file rotation."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: str, log_level: str = "INFO") -> None:
    """Configure console and rotating file handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    
    # Avoid duplicate handlers if this is called twice
    if root_logger.hasHandlers():
        return

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p"
    )

    # Console output (What you see in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File output (For debugging history)
    file_handler = RotatingFileHandler(
        filename=log_path / "app.log", # Path objects work directly in Python 3.6+
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info("Logging configured successfully.")