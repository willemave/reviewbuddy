import json
from pathlib import Path

from app.core.settings import Settings, load_agent_search_env


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
