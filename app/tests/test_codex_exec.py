import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.core.settings import Settings
from app.services.codex_exec import (
    CodexExecError,
    CodexNotInstalledError,
    detect_local_agent_harness,
    run_codex_prompt_sync,
)


class BoolResult(BaseModel):
    ok: bool


def _output_path(command: list[str]) -> Path:
    return Path(command[command.index("--output-last-message") + 1])


def test_run_codex_prompt_sync_structured_success(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001
        assert "--config" in command
        config_index = command.index("--config")
        assert command[config_index + 1] == 'model_reasoning_effort="low"'
        _output_path(command).write_text('{"ok":true}', encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"thread.started","thread_id":"thread-123"}\n'
                '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":4,"cached_input_tokens":3}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, response = run_codex_prompt_sync(
        "Return ok true",
        model_name="gpt-5.4",
        output_type=BoolResult,
    )

    assert result.ok is True
    assert response.thread_id == "thread-123"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 3


def test_run_codex_prompt_sync_schema_mismatch(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001
        _output_path(command).write_text('{"wrong":true}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexExecError):
        run_codex_prompt_sync(
            "Return ok true",
            model_name="gpt-5.4",
            output_type=BoolResult,
        )


def test_run_codex_prompt_sync_nonzero_exit(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexExecError, match="exit code 1"):
        run_codex_prompt_sync("hello", model_name="gpt-5.4")


def test_run_codex_prompt_sync_timeout(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(command, timeout=3)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexExecError, match="timed out"):
        run_codex_prompt_sync("hello", model_name="gpt-5.4", timeout_seconds=3)


def test_run_codex_prompt_sync_missing_binary(monkeypatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001
        raise FileNotFoundError("codex")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(CodexNotInstalledError):
        run_codex_prompt_sync("hello", model_name="gpt-5.4")


def test_run_codex_prompt_sync_auto_detects_claude(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.codex_exec.settings",
        Settings(agent_exec_candidates=["claude", "codex"]),
    )
    monkeypatch.setattr(
        "app.services.codex_exec.shutil.which",
        lambda binary: f"/usr/bin/{binary}" if binary == "claude" else None,
    )

    def fake_run(command, **kwargs):  # noqa: ANN001
        assert command[0] == "claude"
        assert "--output-format" in command
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"type":"assistant","session_id":"session-123","message":{"content":[{"type":"text","text":"{\\"ok\\":true}"}],"usage":{"input_tokens":9,"output_tokens":3,"cache_read_input_tokens":2}}}\n'
                '{"type":"result","subtype":"success","is_error":false,"result":"{\\"ok\\":true}","session_id":"session-123","usage":{"input_tokens":9,"output_tokens":3,"cache_read_input_tokens":2}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, response = run_codex_prompt_sync(
        "Return ok true",
        model_name="claude-sonnet-4-5",
        output_type=BoolResult,
    )

    assert result.ok is True
    assert response.thread_id == "session-123"
    assert response.usage.input_tokens == 9
    assert response.usage.output_tokens == 3
    assert response.usage.cached_input_tokens == 2


def test_run_codex_prompt_sync_uses_custom_command_template(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.codex_exec.settings",
        Settings(
            agent_exec_command_template="printf '{{\"ok\":true}}' > {output_path}",
        ),
    )

    def fake_run(command, **kwargs):  # noqa: ANN001
        assert kwargs["shell"] is True
        assert "{output_path}" not in command
        output_path = command.rsplit(">", maxsplit=1)[1].strip()
        Path(output_path).write_text('{"ok":true}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, response = run_codex_prompt_sync(
        "Return ok true",
        model_name="gpt-5.4",
        output_type=BoolResult,
    )

    assert result.ok is True
    assert response.thread_id is None


def test_detect_local_agent_harness_prefers_available_binary(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.codex_exec.shutil.which",
        lambda binary: f"/usr/bin/{binary}" if binary == "amp" else None,
    )

    resolved = detect_local_agent_harness(Settings(agent_exec_candidates=["amp", "codex"]))

    assert resolved == ("amp", "/usr/bin/amp")
