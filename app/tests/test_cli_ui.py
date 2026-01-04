from pathlib import Path

from rich.console import Console

from app import cli_ui
from app.cli_ui import build_followup_prompt, format_run_config, prompt_config_update
from app.models.review import ReviewRunConfig


def test_build_followup_prompt_includes_sections() -> None:
    prompt = build_followup_prompt("Original", "Report text", "Follow up?")
    assert "Original prompt" in prompt
    assert "Existing report" in prompt
    assert "Follow-up question" in prompt


def test_format_run_config_includes_fields(tmp_path: Path) -> None:
    config = ReviewRunConfig(
        max_urls=12,
        max_agents=4,
        headful=True,
        navigation_timeout_ms=15000,
        output_dir=tmp_path,
        planner_model=None,
        sub_agent_model="gpt-5",
    )
    text = format_run_config(config)
    assert "max_urls: 12" in text
    assert "max_agents: 4" in text
    assert "headful: True" in text
    assert "navigation_timeout_ms: 15000" in text
    assert f"output_dir: {tmp_path}" in text
    assert "planner_model: none" in text
    assert "sub_agent_model: gpt-5" in text


def test_prompt_config_update_applies_changes(monkeypatch, tmp_path: Path) -> None:
    config = ReviewRunConfig(
        max_urls=100,
        max_agents=10,
        headful=True,
        navigation_timeout_ms=20000,
        output_dir=tmp_path,
        planner_model="planner",
        sub_agent_model=None,
    )
    int_values = iter([50, 8, 15000])
    bool_values = iter([False])
    str_values = iter([str(tmp_path / "out"), "none", "gpt-5"])

    monkeypatch.setattr(cli_ui.IntPrompt, "ask", lambda *args, **kwargs: next(int_values))
    monkeypatch.setattr(cli_ui.Confirm, "ask", lambda *args, **kwargs: next(bool_values))
    monkeypatch.setattr(cli_ui.Prompt, "ask", lambda *args, **kwargs: next(str_values))

    console = Console(width=100, record=True)
    updated = prompt_config_update(console, config)

    assert updated.max_urls == 50
    assert updated.max_agents == 8
    assert updated.headful is False
    assert updated.navigation_timeout_ms == 15000
    assert updated.output_dir == Path(tmp_path / "out").expanduser()
    assert updated.planner_model is None
    assert updated.sub_agent_model == "gpt-5"
