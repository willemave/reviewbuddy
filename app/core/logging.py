"""Logging helpers for ReviewBuddy."""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging.

    Args:
        level: Log level name.
    """

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def add_run_file_handler(log_path: Path, level: str = "INFO") -> None:
    """Attach a file handler for a specific run.

    Args:
        log_path: Path to the log file.
        level: Log level name.
    """

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
