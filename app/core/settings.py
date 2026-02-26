"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import APP_NAME, APP_VERSION, DEFAULT_DB_PATH, DEFAULT_STORAGE_DIR


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = APP_NAME
    app_version: str = APP_VERSION

    # Storage
    database_path: Path = DEFAULT_DB_PATH
    storage_path: Path = DEFAULT_STORAGE_DIR

    # External APIs
    exa_api_key: str = ""

    # LLM
    default_model: str = "gpt-5.2"
    planner_model: str = "claude-opus-4-5"
    refiner_model: str = "claude-haiku-4-5"
    synthesizer_model: str = "gemini-3-flash-preview"
    agent_timeout_seconds: int = 120
    agent_temperature: float = 0.6

    # Logging
    log_level: str = "INFO"

    # Crawling
    max_urls: int = Field(default=100, ge=1, le=1000)
    max_agents: int = Field(default=10, ge=1, le=25)
    headful: bool = True
    navigation_timeout_ms: int = Field(default=20000, ge=1000, le=120000)
    crawl_concurrency_per_lane: int = Field(default=6, ge=1, le=20)
    seed_query_budget_ratio: float = Field(default=0.4, ge=0.1, le=0.9)
    refinement_rounds: int = Field(default=3, ge=1, le=6)

    # Exa
    exa_search_type: str = "auto"
    exa_num_results: int = Field(default=30, ge=1, le=100)
    exa_min_results_per_query: int = Field(default=20, ge=1, le=100)
    exa_user_location: str = "US"

    # Google
    google_api_key: str | None = None

    # Reddit
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_username: str | None = None
    reddit_password: str | None = None
    reddit_read_only: bool = True
    reddit_user_agent: str | None = None
    reddit_post_limit: int = Field(default=10, ge=1, le=50)
    reddit_comment_limit: int = Field(default=10, ge=1, le=50)
    reddit_comment_max_chars: int = Field(default=500, ge=100, le=2000)

    # PDF processing
    pdf_model_name: str = "gemini-3-flash-preview"
    pdf_max_bytes: int = Field(default=20_000_000, ge=1, le=200_000_000)
    pdf_summary_max_chars: int = Field(default=6000, ge=500, le=20000)

    # Query shaping
    query_shaping_enabled: bool = True
    query_shaping_suffix: str = (
        'forum OR reddit OR discussion OR "user review" OR blog OR "hands on"'
    )

    # YouTube + Whisper
    youtube_max_videos: int = Field(default=6, ge=0, le=10)
    whisper_model: str = "base"
    whisper_device: str = "auto"
    youtube_summarize_transcripts: bool = True
    youtube_summary_model: str = "gemini-3-flash-preview"
    youtube_summary_concurrency: int = Field(default=3, ge=1, le=10)
    youtube_transcript_max_chars: int = Field(default=2500, ge=500, le=20000)

    # Markdown
    markdown_max_chars: int = Field(default=2500, ge=500, le=20000)
    evidence_max_chars: int = Field(default=1400, ge=200, le=5000)
    markdown_bm25_threshold: float = Field(default=1.0, ge=0.0, le=10.0)
    markdown_pruning_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    markdown_word_count_threshold: int = Field(default=8, ge=1, le=200)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""

    _load_env()
    return Settings()


def _load_env() -> None:
    """Force-load .env from repo root with override."""

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    load_dotenv(env_path, override=True)
