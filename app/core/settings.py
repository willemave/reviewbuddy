"""Application configuration using pydantic-settings."""

import json
import os
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import APP_NAME, APP_VERSION, DEFAULT_DB_PATH, DEFAULT_STORAGE_DIR

SearchProviderName = Literal["exa", "tavily", "firecrawl"]


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
    search_provider: SearchProviderName = "exa"
    search_num_results: int = Field(default=30, ge=1, le=100)
    search_min_results_per_query: int = Field(default=20, ge=1, le=100)
    exa_api_key: str = ""
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""

    # LLM
    default_model: str = "gpt-5.4"
    planner_model: str = "gpt-5.4"
    refiner_model: str = "gpt-5.4"
    synthesizer_model: str = "gpt-5.4"
    agent_timeout_seconds: int = 120
    agent_temperature: float = 0.6
    agent_exec_path: str = ""
    agent_exec_candidates: list[str] = Field(default_factory=lambda: ["codex", "claude", "amp"])
    agent_exec_command_template: str = ""
    agent_exec_sandbox: str = "read-only"
    agent_exec_model_reasoning_effort: str = "low"
    agent_exec_extra_args: list[str] = Field(default_factory=list)
    codex_exec_path: str = "codex"
    codex_exec_sandbox: str = "read-only"
    codex_exec_model_reasoning_effort: str = "low"
    codex_exec_extra_args: list[str] = Field(default_factory=list)
    rlm_root_model: str = "gpt-5.4"
    rlm_subquery_model: str = "gpt-5.4"
    rlm_max_iterations: int = Field(default=4, ge=1, le=20)

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
    exa_user_location: str = "US"

    # Tavily
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    tavily_topic: Literal["general", "news", "finance"] = "general"
    tavily_auto_parameters: bool = False
    tavily_max_results: int = Field(default=20, ge=1, le=20)

    # Firecrawl
    firecrawl_country: str = "US"
    firecrawl_location: str | None = "United States"

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
    pdf_model_name: str = "gpt-5.4"
    pdf_max_bytes: int = Field(default=20_000_000, ge=1, le=200_000_000)
    pdf_input_max_chars: int = Field(default=40000, ge=1000, le=200000)
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
    youtube_summary_model: str = "gpt-5.4"
    youtube_summary_concurrency: int = Field(default=3, ge=1, le=10)
    youtube_transcript_max_chars: int = Field(default=2500, ge=500, le=20000)
    youtube_ingest_timeout_seconds: int = Field(default=45, ge=5, le=600)

    # Markdown
    markdown_max_chars: int = Field(default=2500, ge=500, le=20000)
    evidence_max_chars: int = Field(default=1400, ge=200, le=5000)
    markdown_bm25_threshold: float = Field(default=1.0, ge=0.0, le=10.0)
    markdown_pruning_threshold: float = Field(default=0.48, ge=0.0, le=1.0)
    markdown_word_count_threshold: int = Field(default=8, ge=1, le=200)
    synthesis_merge_target_tokens: int = Field(default=180000, ge=1000, le=200000)
    synthesis_merge_hard_max_tokens: int = Field(default=200000, ge=2000, le=250000)
    synthesis_merge_max_children: int = Field(default=4, ge=2, le=16)
    synthesis_merge_max_sources: int = Field(default=12, ge=1, le=50)
    synthesis_final_target_tokens: int = Field(default=180000, ge=2000, le=250000)
    synthesis_final_hard_max_tokens: int = Field(default=200000, ge=4000, le=400000)
    synthesis_final_max_sources: int = Field(default=18, ge=1, le=80)

    def get_effective_search_provider(self) -> SearchProviderName:
        """Return the active search provider, auto-selecting from configured keys when possible."""

        if "search_provider" in self.model_fields_set:
            return self.search_provider

        detected = self.detect_search_provider_from_keys()
        if detected is not None:
            return detected
        return self.search_provider

    def get_search_provider_key_name(self, provider: SearchProviderName | None = None) -> str:
        """Return the environment variable name for a search provider API key."""

        selected_provider = provider or self.get_effective_search_provider()
        return {
            "exa": "EXA_API_KEY",
            "tavily": "TAVILY_API_KEY",
            "firecrawl": "FIRECRAWL_API_KEY",
        }[selected_provider]

    def get_search_provider_api_key(self, provider: SearchProviderName | None = None) -> str:
        """Return the configured API key for a search provider."""

        selected_provider = provider or self.get_effective_search_provider()
        return {
            "exa": self.exa_api_key,
            "tavily": self.tavily_api_key,
            "firecrawl": self.firecrawl_api_key,
        }[selected_provider].strip()

    def detect_search_provider_from_keys(self) -> SearchProviderName | None:
        """Return the first provider that has a configured API key."""

        if self.exa_api_key.strip():
            return "exa"
        if self.tavily_api_key.strip():
            return "tavily"
        if self.firecrawl_api_key.strip():
            return "firecrawl"
        return None


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
    load_agent_search_env(Path.home())


def load_agent_search_env(
    home_dir: Path,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Load search-provider settings from Hermes and OpenClaw config files.

    Args:
        home_dir: User home directory.
        env: Mapping to update. Defaults to `os.environ`.
    """

    target_env = env if env is not None else os.environ
    _load_hermes_env(home_dir / ".hermes" / ".env", target_env)
    _load_openclaw_config(home_dir / ".openclaw" / "openclaw.json", target_env)


def _load_hermes_env(env_path: Path, target_env: MutableMapping[str, str]) -> None:
    if not env_path.exists():
        return

    values = dotenv_values(env_path)
    for key in (
        "SEARCH_PROVIDER",
        "EXA_API_KEY",
        "EXA_SEARCH_TYPE",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        value = values.get(key)
        if isinstance(value, str) and value.strip() and key not in target_env:
            target_env[key] = value.strip()


def _load_openclaw_config(config_path: Path, target_env: MutableMapping[str, str]) -> None:
    if not config_path.exists():
        return

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    tools = payload.get("tools")
    if not isinstance(tools, dict):
        return
    web = tools.get("web")
    if not isinstance(web, dict):
        return
    search = web.get("search")
    if not isinstance(search, dict):
        return

    provider = search.get("provider")
    if provider not in {"exa", "tavily", "firecrawl"}:
        return

    provider_config = search.get(provider)
    if not isinstance(provider_config, dict):
        return

    api_key = provider_config.get("apiKey")
    if not isinstance(api_key, str) or not api_key.strip():
        return

    target_env.setdefault("SEARCH_PROVIDER", provider)
    target_env.setdefault(
        {
            "exa": "EXA_API_KEY",
            "tavily": "TAVILY_API_KEY",
            "firecrawl": "FIRECRAWL_API_KEY",
        }[provider],
        api_key.strip(),
    )

    if provider == "exa":
        search_type = provider_config.get("type")
        if isinstance(search_type, str) and search_type.strip():
            target_env.setdefault("EXA_SEARCH_TYPE", search_type.strip())
