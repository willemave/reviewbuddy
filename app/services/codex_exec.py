"""Codex exec bridge for structured and text LLM calls."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.settings import get_settings

settings = get_settings()

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CodexUsage:
    """Usage counters parsed from `codex exec` JSON events."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return total input and output tokens."""

        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CodexResponse:
    """Raw response metadata from a Codex exec invocation."""

    message: str
    usage: CodexUsage
    thread_id: str | None = None
    stdout: str = ""
    stderr: str = ""


class CodexExecError(RuntimeError):
    """Raised when Codex exec fails."""


class CodexNotInstalledError(CodexExecError):
    """Raised when the Codex binary is not installed."""


def run_codex_prompt_sync(
    prompt: str,
    *,
    model_name: str | None = None,
    output_type: type[T] | None = None,
    timeout_seconds: int | None = None,
    cwd: Path | None = None,
    resume_session_id: str | None = None,
) -> CodexResponse | tuple[T, CodexResponse]:
    """Run Codex synchronously and optionally validate structured output."""

    response = _run_codex_prompt(
        prompt=prompt,
        model_name=model_name,
        output_type=output_type,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
        resume_session_id=resume_session_id,
    )
    if output_type is None:
        return response
    return _validate_output(response.message, output_type), response


async def run_codex_prompt(
    prompt: str,
    *,
    model_name: str | None = None,
    output_type: type[T] | None = None,
    timeout_seconds: int | None = None,
    cwd: Path | None = None,
    resume_session_id: str | None = None,
) -> CodexResponse | tuple[T, CodexResponse]:
    """Run Codex asynchronously and optionally validate structured output."""

    response = await asyncio.to_thread(
        _run_codex_prompt,
        prompt=prompt,
        model_name=model_name,
        output_type=output_type,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
        resume_session_id=resume_session_id,
    )
    if output_type is None:
        return response
    return _validate_output(response.message, output_type), response


def _run_codex_prompt(
    *,
    prompt: str,
    model_name: str | None,
    output_type: type[T] | None,
    timeout_seconds: int | None,
    cwd: Path | None,
    resume_session_id: str | None,
) -> CodexResponse:
    command, output_path, schema_path = _build_command(
        prompt=prompt,
        model_name=model_name,
        output_type=output_type,
        resume_session_id=resume_session_id,
    )
    exec_cwd = str(cwd or Path.cwd())
    effective_timeout = timeout_seconds or settings.agent_timeout_seconds

    try:
        completed = subprocess.run(
            command,
            cwd=exec_cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except FileNotFoundError as exc:
        raise CodexNotInstalledError(
            f"Codex executable not found: {settings.codex_exec_path}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CodexExecError(
            f"Codex exec timed out after {effective_timeout} seconds"
        ) from exc
    finally:
        if schema_path is not None:
            _safe_unlink(schema_path)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        _safe_unlink(output_path)
        raise CodexExecError(
            f"Codex exec failed with exit code {completed.returncode}: {stderr.strip() or stdout.strip()}"
        )

    message = _read_output_message(output_path, stdout)
    usage, thread_id = _parse_exec_events(stdout)
    _safe_unlink(output_path)
    return CodexResponse(
        message=message,
        usage=usage,
        thread_id=thread_id,
        stdout=stdout,
        stderr=stderr,
    )


def _build_command(
    *,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    resume_session_id: str | None,
) -> tuple[list[str], str, str | None]:
    with tempfile.NamedTemporaryFile(
        prefix="reviewbuddy-codex-",
        suffix=".txt",
        delete=False,
    ) as output_file:
        output_path = output_file.name

    model = model_name or settings.default_model
    base_command = [settings.codex_exec_path, "exec"]
    schema_path: str | None = None

    if resume_session_id:
        if output_type is not None:
            _safe_unlink(output_path)
            raise CodexExecError("Structured output is not supported when resuming Codex sessions")
        base_command.extend(["resume"])
        option_args = ["--json", "--output-last-message", output_path]
        if model:
            option_args.extend(["--model", model])
        command = [*base_command, *option_args, *_common_flags(), resume_session_id, prompt]
        return command, output_path, None

    option_args = ["--json", "--output-last-message", output_path]
    if model:
        option_args.extend(["--model", model])
    if output_type is not None:
        with tempfile.NamedTemporaryFile(
            prefix="reviewbuddy-codex-schema-",
            suffix=".json",
            delete=False,
        ) as schema_file:
            schema_path = schema_file.name
            schema = _normalize_json_schema(output_type.model_json_schema())
            schema_file.write(json.dumps(schema, ensure_ascii=True).encode("utf-8"))
        option_args.extend(["--output-schema", schema_path])

    return [*base_command, *option_args, *_common_flags(), prompt], output_path, schema_path


def _common_flags() -> list[str]:
    flags = [
        "--skip-git-repo-check",
        "--sandbox",
        settings.codex_exec_sandbox,
    ]
    if settings.codex_exec_model_reasoning_effort:
        flags.extend(
            [
                "--config",
                f'model_reasoning_effort="{settings.codex_exec_model_reasoning_effort}"',
            ]
        )
    flags.extend(settings.codex_exec_extra_args)
    return flags


def _read_output_message(output_path: str, stdout: str) -> str:
    try:
        return Path(output_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass

    for line in stdout.splitlines():
        text = _extract_agent_message(line)
        if text is not None:
            return text
    raise CodexExecError("Codex exec completed without a final message")


def _parse_exec_events(stdout: str) -> tuple[CodexUsage, str | None]:
    usage = CodexUsage(requests=1)
    thread_id: str | None = None

    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = payload.get("type")
        if event_type == "thread.started":
            maybe_thread = payload.get("thread_id")
            if isinstance(maybe_thread, str):
                thread_id = maybe_thread
            continue

        if event_type != "turn.completed":
            continue

        usage_payload = payload.get("usage", {})
        usage = CodexUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cached_input_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
            requests=1,
        )

    return usage, thread_id


def _extract_agent_message(line: str) -> str | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    if payload.get("type") != "item.completed":
        return None
    item = payload.get("item", {})
    if item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text if isinstance(text, str) else None


def _validate_output(message: str, output_type: type[T]) -> T:
    try:
        return output_type.model_validate_json(message)
    except ValidationError as exc:
        raise CodexExecError(f"Codex output did not match schema for {output_type.__name__}") from exc
    except json.JSONDecodeError as exc:
        raise CodexExecError(f"Codex output was not valid JSON for {output_type.__name__}") from exc


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _normalize_json_schema(node: object) -> object:
    if isinstance(node, dict):
        normalized = {key: _normalize_json_schema(value) for key, value in node.items()}
        if normalized.get("type") == "object":
            normalized.setdefault("additionalProperties", False)
            properties = normalized.get("properties")
            if isinstance(properties, dict):
                normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(node, list):
        return [_normalize_json_schema(item) for item in node]
    return node
