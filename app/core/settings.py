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

from app.constants import APP_NAME, APP_VERSION, default_data_dir

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
    database_path: Path = Field(default_factory=lambda: default_data_dir() / "researchbuddy.db")
    storage_path: Path = Field(default_factory=lambda: default_data_dir() / "storage")

    # External APIs
    search_provider: SearchProviderName = "exa"
    search_num_results: int = Field(default=20, ge=1, le=100)
    search_min_results_per_query: int = Field(default=10, ge=1, le=100)
    search_query_budget: int = Field(default=40, ge=1, le=200)
    search_query_budget_per_lane: int = Field(default=5, ge=1, le=20)
    initial_search_queries_per_lane: int = Field(default=3, ge=1, le=10)
    semantic_dedupe_enabled: bool = True
    semantic_embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    semantic_embedding_device: str = "auto"
    semantic_embedding_local_files_only: bool = True
    semantic_embedding_max_length: int = Field(default=512, ge=32, le=32768)
    semantic_embedding_batch_size: int = Field(default=16, ge=1, le=256)
    semantic_query_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    semantic_lane_similarity_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    semantic_card_mmr_lambda: float = Field(default=0.72, ge=0.0, le=1.0)
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
    query_shaping_suffix: str = 'forum OR reddit OR discussion OR blog OR "hands on"'

    # YouTube + Whisper
    youtube_max_videos: int = Field(default=6, ge=0, le=10)
    whisper_model: str = "base"
    whisper_device: str = "auto"
    youtube_summarize_transcripts: bool = True
    youtube_summary_model: str = "gpt-5.4"
    youtube_summary_concurrency: int = Field(default=3, ge=1, le=10)
    youtube_transcript_max_chars: int = Field(default=2500, ge=500, le=20000)
    youtube_ingest_timeout_seconds: int = Field(default=45, ge=5, le=600)

    # Podcasts + Whisper
    podcast_max_episodes: int = Field(default=4, ge=0, le=10)
    podcast_summarize_transcripts: bool = True
    podcast_summary_model: str = "gpt-5.4"
    podcast_summary_concurrency: int = Field(default=3, ge=1, le=10)
    podcast_transcript_max_chars: int = Field(default=2500, ge=500, le=20000)
    podcast_ingest_timeout_seconds: int = Field(default=90, ge=5, le=900)

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
    """Load shared agent config first, then local .env defaults."""

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    load_agent_search_env(Path.home())
    load_dotenv(env_path, override=False)


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
    _load_openclaw_env(home_dir / ".openclaw" / ".env", target_env)
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
            cleaned = value.strip()
            if key == "SEARCH_PROVIDER" and cleaned not in {"exa", "tavily", "firecrawl"}:
                continue
            target_env[key] = cleaned


def _load_openclaw_env(env_path: Path, target_env: MutableMapping[str, str]) -> None:
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
            cleaned = value.strip()
            if key == "SEARCH_PROVIDER" and cleaned not in {"exa", "tavily", "firecrawl"}:
                continue
            target_env[key] = cleaned


def _load_openclaw_config(config_path: Path, target_env: MutableMapping[str, str]) -> None:
    if not config_path.exists():
        return

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    resolved = _resolve_openclaw_search_provider(payload, target_env)
    if resolved is None:
        return

    provider, api_key = resolved
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
        search_type = _resolve_openclaw_exa_search_type(payload)
        if isinstance(search_type, str) and search_type.strip():
            target_env.setdefault("EXA_SEARCH_TYPE", search_type.strip())


def _resolve_openclaw_search_provider(
    payload: dict,
    target_env: MutableMapping[str, str],
) -> tuple[SearchProviderName, str] | None:
    explicit_provider = _read_nested(payload, ("tools", "web", "search", "provider"))
    if explicit_provider in {"exa", "tavily", "firecrawl"}:
        provider_order: tuple[SearchProviderName, ...] = (explicit_provider,)
    else:
        provider_order = ("exa", "tavily", "firecrawl")

    for provider in provider_order:
        api_key = _resolve_openclaw_provider_api_key(payload, provider, target_env)
        if api_key:
            return provider, api_key
    return None


def _resolve_openclaw_provider_api_key(
    payload: dict,
    provider: SearchProviderName,
    target_env: MutableMapping[str, str],
) -> str | None:
    candidates = [
        ("plugins", "entries", provider, "config", "webSearch", "apiKey"),
        ("tools", "web", "search", provider, "apiKey"),
    ]
    if provider == "firecrawl":
        candidates.extend(
            [
                ("plugins", "entries", "firecrawl", "config", "webFetch", "apiKey"),
                ("tools", "web", "fetch", "firecrawl", "apiKey"),
            ]
        )
    if _read_nested(payload, ("tools", "web", "search", "provider")) == provider:
        candidates.append(("tools", "web", "search", "apiKey"))

    for path in candidates:
        api_key = _resolve_secret_value(_read_nested(payload, path), target_env)
        if api_key:
            return api_key
    return None


def _resolve_openclaw_exa_search_type(payload: dict) -> str | None:
    candidates = (
        ("plugins", "entries", "exa", "config", "webSearch", "type"),
        ("tools", "web", "search", "exa", "type"),
    )
    for path in candidates:
        value = _read_nested(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_nested(payload: object, path: tuple[str, ...]) -> object:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_secret_value(
    value: object,
    target_env: MutableMapping[str, str],
) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.startswith("secretref-env:"):
            return _resolve_env_secret(cleaned.removeprefix("secretref-env:"), target_env)
        return cleaned
    if not isinstance(value, dict):
        return None
    if value.get("source") != "env":
        return None
    env_name = value.get("id")
    if not isinstance(env_name, str):
        return None
    return _resolve_env_secret(env_name, target_env)


def _resolve_env_secret(env_name: str, target_env: MutableMapping[str, str]) -> str | None:
    cleaned_name = env_name.strip()
    if not cleaned_name:
        return None
    value = target_env.get(cleaned_name) or os.environ.get(cleaned_name)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
