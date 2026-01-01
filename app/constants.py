"""Application constants."""

from pathlib import Path

APP_NAME = "Review Buddy"
APP_VERSION = "0.1.0"
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR / "storage"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "reviewbuddy.db"

URL_STATUS_PENDING = "pending"
URL_STATUS_FETCHED = "fetched"
URL_STATUS_FAILED = "failed"

RUN_STATUS_IN_PROGRESS = "in_progress"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
