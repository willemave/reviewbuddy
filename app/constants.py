"""Application constants."""

import os
from pathlib import Path

APP_NAME = "ResearchBuddy"
APP_VERSION = "0.1.7"


def default_data_dir() -> Path:
    """Return the default per-user ResearchBuddy state directory."""

    configured = os.environ.get("RESEARCHBUDDY_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".researchbuddy"


DEFAULT_DATA_DIR = default_data_dir()
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR / "storage"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "researchbuddy.db"

URL_STATUS_PENDING = "pending"
URL_STATUS_FETCHED = "fetched"
URL_STATUS_FAILED = "failed"

RUN_STATUS_IN_PROGRESS = "in_progress"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"

FOLLOWUP_MEMORY_FILENAME = "followup_memory.json"
YOUTUBE_TRANSCRIPTS_FILENAME = "youtube_transcripts.json"
PODCAST_TRANSCRIPTS_FILENAME = "podcast_transcripts.json"
