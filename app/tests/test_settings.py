import json
from pathlib import Path

from app.core.settings import Settings, load_agent_search_env


def test_settings_defaults_to_user_researchbuddy_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RESEARCHBUDDY_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.delenv("STORAGE_PATH", raising=False)

    settings = Settings()

    assert settings.database_path == tmp_path / "state" / "researchbuddy.db"
    assert settings.storage_path == tmp_path / "state" / "storage"


def test_load_agent_search_env_reads_hermes_env(tmp_path: Path) -> None:
    hermes_dir = tmp_path / ".hermes"
    hermes_dir.mkdir()
    (hermes_dir / ".env").write_text(
        "TAVILY_API_KEY=hermes-tavily\nSEARCH_PROVIDER=tavily\n",
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    load_agent_search_env(tmp_path, env)

    assert env["TAVILY_API_KEY"] == "hermes-tavily"
    assert env["SEARCH_PROVIDER"] == "tavily"


def test_load_agent_search_env_reads_openclaw_config(tmp_path: Path) -> None:
    openclaw_dir = tmp_path / ".openclaw"
    openclaw_dir.mkdir()
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps(
            {
                "tools": {
                    "web": {
                        "search": {
                            "provider": "exa",
                            "exa": {
                                "apiKey": "openclaw-exa",
                                "type": "auto",
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    load_agent_search_env(tmp_path, env)

    assert env["SEARCH_PROVIDER"] == "exa"
    assert env["EXA_API_KEY"] == "openclaw-exa"
    assert env["EXA_SEARCH_TYPE"] == "auto"


def test_load_agent_search_env_reads_openclaw_plugin_config(tmp_path: Path) -> None:
    openclaw_dir = tmp_path / ".openclaw"
    openclaw_dir.mkdir()
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "tavily": {
                            "config": {
                                "webSearch": {
                                    "apiKey": "plugin-tavily",
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    load_agent_search_env(tmp_path, env)

    assert env["SEARCH_PROVIDER"] == "tavily"
    assert env["TAVILY_API_KEY"] == "plugin-tavily"


def test_load_agent_search_env_resolves_openclaw_env_secret_refs(tmp_path: Path) -> None:
    openclaw_dir = tmp_path / ".openclaw"
    openclaw_dir.mkdir()
    (openclaw_dir / ".env").write_text("FIRECRAWL_API_KEY=from-openclaw-env\n", encoding="utf-8")
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "firecrawl": {
                            "config": {
                                "webSearch": {
                                    "apiKey": {
                                        "source": "env",
                                        "provider": "default",
                                        "id": "FIRECRAWL_API_KEY",
                                    },
                                },
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    load_agent_search_env(tmp_path, env)

    assert env["SEARCH_PROVIDER"] == "firecrawl"
    assert env["FIRECRAWL_API_KEY"] == "from-openclaw-env"


def test_load_agent_search_env_preserves_existing_values(tmp_path: Path) -> None:
    openclaw_dir = tmp_path / ".openclaw"
    openclaw_dir.mkdir()
    (openclaw_dir / "openclaw.json").write_text(
        json.dumps(
            {
                "tools": {
                    "web": {
                        "search": {
                            "provider": "exa",
                            "exa": {
                                "apiKey": "openclaw-exa",
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env = {
        "SEARCH_PROVIDER": "tavily",
        "EXA_API_KEY": "existing-exa",
    }
    load_agent_search_env(tmp_path, env)

    assert env["SEARCH_PROVIDER"] == "tavily"
    assert env["EXA_API_KEY"] == "existing-exa"


def test_settings_prefers_explicit_search_provider_override() -> None:
    settings = Settings(
        search_provider="firecrawl",
        exa_api_key="exa-key",
        firecrawl_api_key="fc-key",
    )

    assert settings.get_effective_search_provider() == "firecrawl"


def test_settings_use_lower_search_defaults() -> None:
    settings = Settings()

    assert settings.search_num_results == 20
    assert settings.search_min_results_per_query == 10
    assert settings.search_query_budget == 40
    assert settings.semantic_dedupe_enabled is True
    assert settings.semantic_embedding_model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.semantic_embedding_local_files_only is True
    assert settings.semantic_query_similarity_threshold == 0.92
    assert settings.semantic_lane_similarity_threshold == 0.88
    assert settings.semantic_card_mmr_lambda == 0.72
