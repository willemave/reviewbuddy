import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.services.codex_exec import (
    CodexExecError,
    CodexNotInstalledError,
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
