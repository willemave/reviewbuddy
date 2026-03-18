"""Local coding-agent exec bridge for structured and text LLM calls."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.settings import Settings, get_settings

settings = get_settings()

T = TypeVar("T", bound=BaseModel)

SUPPORTED_LOCAL_AGENT_HARNESSES = frozenset({"codex", "claude", "amp"})


@dataclass(frozen=True)
class CodexUsage:
    """Usage counters parsed from a local agent harness response."""

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
    """Raw response metadata from a local agent harness invocation."""

    message: str
    usage: CodexUsage
    thread_id: str | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class LocalAgentHarness:
    """Resolved local coding-agent harness configuration."""

    name: str
    executable: str
    command_template: str | None = None


class CodexExecError(RuntimeError):
    """Raised when local agent execution fails."""


class CodexNotInstalledError(CodexExecError):
    """Raised when no supported local agent binary is installed."""


def run_codex_prompt_sync(
    prompt: str,
    *,
    model_name: str | None = None,
    output_type: type[T] | None = None,
    timeout_seconds: int | None = None,
    cwd: Path | None = None,
    resume_session_id: str | None = None,
) -> CodexResponse | tuple[T, CodexResponse]:
    """Run a local coding agent synchronously and optionally validate structured output."""

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
    """Run a local coding agent asynchronously and optionally validate structured output."""

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


def detect_local_agent_harness(
    configured_settings: Settings | None = None,
) -> tuple[str, str] | None:
    """Return the selected local agent harness name and executable path."""

    settings_obj = configured_settings or settings
    harness = _resolve_local_agent_harness(settings_obj)
    resolved_path = shutil.which(harness.executable)
    if resolved_path:
        return harness.name, resolved_path
    if Path(harness.executable).exists():
        return harness.name, harness.executable
    if harness.command_template:
        return harness.name, harness.executable
    return None


def _run_codex_prompt(
    *,
    prompt: str,
    model_name: str | None,
    output_type: type[T] | None,
    timeout_seconds: int | None,
    cwd: Path | None,
    resume_session_id: str | None,
) -> CodexResponse:
    harness = _resolve_local_agent_harness(settings)
    exec_cwd = str(cwd or Path.cwd())
    effective_timeout = timeout_seconds or settings.agent_timeout_seconds

    if harness.command_template:
        return _run_custom_harness(
            harness=harness,
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            timeout_seconds=effective_timeout,
            cwd=exec_cwd,
            resume_session_id=resume_session_id,
        )
    if harness.name == "codex":
        return _run_codex_harness(
            harness=harness,
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            timeout_seconds=effective_timeout,
            cwd=exec_cwd,
            resume_session_id=resume_session_id,
        )
    if harness.name == "claude":
        return _run_claude_harness(
            harness=harness,
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            timeout_seconds=effective_timeout,
            cwd=exec_cwd,
            resume_session_id=resume_session_id,
        )
    if harness.name == "amp":
        return _run_amp_harness(
            harness=harness,
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            timeout_seconds=effective_timeout,
            cwd=exec_cwd,
            resume_session_id=resume_session_id,
        )

    raise CodexExecError(f"Unsupported local agent harness: {harness.name}")


def _run_codex_harness(
    *,
    harness: LocalAgentHarness,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    timeout_seconds: int,
    cwd: str,
    resume_session_id: str | None,
) -> CodexResponse:
    command, output_path, schema_path = _build_codex_command(
        harness=harness,
        prompt=prompt,
        model_name=model_name,
        output_type=output_type,
        resume_session_id=resume_session_id,
    )

    try:
        completed = _run_subprocess(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
    finally:
        if schema_path is not None:
            _safe_unlink(schema_path)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        _safe_unlink(output_path)
        raise CodexExecError(
            f"Local agent exec failed with exit code {completed.returncode}: {stderr.strip() or stdout.strip()}"
        )

    message = _read_output_message(output_path, stdout)
    usage, thread_id = _parse_codex_exec_events(stdout)
    _safe_unlink(output_path)
    return CodexResponse(
        message=message,
        usage=usage,
        thread_id=thread_id,
        stdout=stdout,
        stderr=stderr,
    )


def _run_claude_harness(
    *,
    harness: LocalAgentHarness,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    timeout_seconds: int,
    cwd: str,
    resume_session_id: str | None,
) -> CodexResponse:
    command = _build_claude_command(
        harness=harness,
        prompt=prompt,
        model_name=model_name,
        output_type=output_type,
        resume_session_id=resume_session_id,
    )
    completed = _run_subprocess(
        command=command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        raise CodexExecError(
            f"Local agent exec failed with exit code {completed.returncode}: {stderr.strip() or stdout.strip()}"
        )
    message, usage, thread_id = _parse_stream_json_result(stdout)
    return CodexResponse(
        message=message,
        usage=usage,
        thread_id=thread_id,
        stdout=stdout,
        stderr=stderr,
    )


def _run_amp_harness(
    *,
    harness: LocalAgentHarness,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    timeout_seconds: int,
    cwd: str,
    resume_session_id: str | None,
) -> CodexResponse:
    del model_name

    if resume_session_id:
        raise CodexExecError("Resuming local agent sessions is not supported for amp")

    final_prompt = prompt
    if output_type is not None:
        final_prompt = _augment_prompt_for_json(prompt, output_type)

    command = [
        harness.executable,
        "--execute",
        final_prompt,
        "--stream-json",
        *_resolved_extra_args(settings),
    ]
    completed = _run_subprocess(
        command=command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        raise CodexExecError(
            f"Local agent exec failed with exit code {completed.returncode}: {stderr.strip() or stdout.strip()}"
        )
    message, usage, thread_id = _parse_stream_json_result(stdout)
    return CodexResponse(
        message=message,
        usage=usage,
        thread_id=thread_id,
        stdout=stdout,
        stderr=stderr,
    )


def _run_custom_harness(
    *,
    harness: LocalAgentHarness,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    timeout_seconds: int,
    cwd: str,
    resume_session_id: str | None,
) -> CodexResponse:
    if resume_session_id:
        raise CodexExecError("Resuming local agent sessions is only supported for codex and claude")

    final_prompt = prompt
    schema_path: str | None = None
    if output_type is not None:
        final_prompt = _augment_prompt_for_json(prompt, output_type)
        schema_path = _write_json_schema(output_type, prefix="reviewbuddy-agent-schema-")

    with tempfile.NamedTemporaryFile(
        prefix="reviewbuddy-agent-prompt-",
        suffix=".txt",
        delete=False,
    ) as prompt_file:
        prompt_path = prompt_file.name
        prompt_file.write(final_prompt.encode("utf-8"))

    with tempfile.NamedTemporaryFile(
        prefix="reviewbuddy-agent-output-",
        suffix=".txt",
        delete=False,
    ) as output_file:
        output_path = output_file.name

    command = _render_custom_command(
        harness.command_template or "",
        prompt=final_prompt,
        prompt_path=prompt_path,
        output_path=output_path,
        schema_path=schema_path,
        model_name=model_name,
        cwd=cwd,
    )

    try:
        completed = _run_subprocess(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            shell=True,
        )
    finally:
        _safe_unlink(prompt_path)
        _safe_unlink(schema_path)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        _safe_unlink(output_path)
        raise CodexExecError(
            f"Local agent exec failed with exit code {completed.returncode}: {stderr.strip() or stdout.strip()}"
        )

    message = _read_output_message(output_path, stdout)
    _safe_unlink(output_path)
    return CodexResponse(
        message=message,
        usage=CodexUsage(requests=1),
        thread_id=None,
        stdout=stdout,
        stderr=stderr,
    )


def _build_codex_command(
    *,
    harness: LocalAgentHarness,
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
    base_command = [harness.executable, "exec"]
    schema_path: str | None = None

    if resume_session_id:
        if output_type is not None:
            _safe_unlink(output_path)
            raise CodexExecError(
                "Structured output is not supported when resuming local agent sessions"
            )
        base_command.extend(["resume"])
        option_args = ["--json", "--output-last-message", output_path]
        if model:
            option_args.extend(["--model", model])
        command = [
            *base_command,
            *option_args,
            *_codex_common_flags(),
            resume_session_id,
            prompt,
        ]
        return command, output_path, None

    option_args = ["--json", "--output-last-message", output_path]
    if model:
        option_args.extend(["--model", model])
    if output_type is not None:
        schema_path = _write_json_schema(output_type, prefix="reviewbuddy-codex-schema-")
        option_args.extend(["--output-schema", schema_path])

    return [*base_command, *option_args, *_codex_common_flags(), prompt], output_path, schema_path


def _build_claude_command(
    *,
    harness: LocalAgentHarness,
    prompt: str,
    model_name: str | None,
    output_type: type[BaseModel] | None,
    resume_session_id: str | None,
) -> list[str]:
    if output_type is not None and resume_session_id:
        raise CodexExecError(
            "Structured output is not supported when resuming local agent sessions"
        )

    command = [harness.executable]
    if resume_session_id:
        command.extend(["--resume", resume_session_id])
    command.extend(["--print", prompt, "--output-format", "stream-json"])
    if model_name:
        command.extend(["--model", model_name])
    if output_type is not None:
        schema = _normalize_json_schema(output_type.model_json_schema())
        command.extend(["--json-schema", json.dumps(schema, ensure_ascii=True)])
    command.extend(_resolved_extra_args(settings))
    return command


def _run_subprocess(
    *,
    command: list[str] | str,
    cwd: str,
    timeout_seconds: int,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=shell,
        )
    except FileNotFoundError as exc:
        raise CodexNotInstalledError("No supported local agent executable found") from exc
    except subprocess.TimeoutExpired as exc:
        raise CodexExecError(f"Local agent exec timed out after {timeout_seconds} seconds") from exc


def _resolve_local_agent_harness(settings_obj: Settings) -> LocalAgentHarness:
    command_template = settings_obj.agent_exec_command_template.strip()
    if command_template:
        executable = _extract_command_executable(command_template)
        if executable is None:
            return LocalAgentHarness(
                name="custom",
                executable="custom",
                command_template=command_template,
            )
        return LocalAgentHarness(
            name=_normalize_harness_name(executable),
            executable=executable,
            command_template=command_template,
        )

    preferred_executable = _resolved_exec_path(settings_obj)
    if preferred_executable:
        harness_name = _normalize_harness_name(preferred_executable)
        if harness_name not in SUPPORTED_LOCAL_AGENT_HARNESSES:
            raise CodexExecError(
                "Unsupported local agent harness. Configure AGENT_EXEC_COMMAND_TEMPLATE for custom CLIs."
            )
        return LocalAgentHarness(name=harness_name, executable=preferred_executable)

    for candidate in settings_obj.agent_exec_candidates:
        normalized = _normalize_harness_name(candidate)
        if normalized not in SUPPORTED_LOCAL_AGENT_HARNESSES:
            continue
        resolved_path = shutil.which(candidate)
        if resolved_path:
            return LocalAgentHarness(name=normalized, executable=candidate)

    searched = ", ".join(settings_obj.agent_exec_candidates)
    raise CodexNotInstalledError(f"No supported local agent executable found. Checked: {searched}")


def _resolved_exec_path(settings_obj: Settings) -> str | None:
    if settings_obj.agent_exec_path.strip():
        return settings_obj.agent_exec_path.strip()
    if _field_is_overridden(settings_obj, "codex_exec_path"):
        return settings_obj.codex_exec_path.strip()
    return None


def _resolved_extra_args(settings_obj: Settings) -> list[str]:
    if _field_is_overridden(settings_obj, "agent_exec_extra_args"):
        return settings_obj.agent_exec_extra_args
    if _field_is_overridden(settings_obj, "codex_exec_extra_args"):
        return settings_obj.codex_exec_extra_args
    return settings_obj.agent_exec_extra_args


def _resolved_sandbox(settings_obj: Settings) -> str:
    if _field_is_overridden(settings_obj, "agent_exec_sandbox"):
        return settings_obj.agent_exec_sandbox
    if _field_is_overridden(settings_obj, "codex_exec_sandbox"):
        return settings_obj.codex_exec_sandbox
    return settings_obj.agent_exec_sandbox


def _resolved_model_reasoning_effort(settings_obj: Settings) -> str:
    if _field_is_overridden(settings_obj, "agent_exec_model_reasoning_effort"):
        return settings_obj.agent_exec_model_reasoning_effort
    if _field_is_overridden(settings_obj, "codex_exec_model_reasoning_effort"):
        return settings_obj.codex_exec_model_reasoning_effort
    return settings_obj.agent_exec_model_reasoning_effort


def _codex_common_flags() -> list[str]:
    flags = [
        "--skip-git-repo-check",
        "--sandbox",
        _resolved_sandbox(settings),
    ]
    reasoning_effort = _resolved_model_reasoning_effort(settings)
    if reasoning_effort:
        flags.extend(
            [
                "--config",
                f'model_reasoning_effort="{reasoning_effort}"',
            ]
        )
    flags.extend(_resolved_extra_args(settings))
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
    if stdout.strip():
        return stdout.strip()
    raise CodexExecError("Local agent exec completed without a final message")


def _parse_codex_exec_events(stdout: str) -> tuple[CodexUsage, str | None]:
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
        usage = _usage_from_payload(usage_payload)

    return usage, thread_id


def _parse_stream_json_result(stdout: str) -> tuple[str, CodexUsage, str | None]:
    usage = CodexUsage(requests=1)
    session_id: str | None = None
    result_text: str | None = None
    last_text_message: str | None = None

    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        maybe_session_id = payload.get("session_id")
        if isinstance(maybe_session_id, str):
            session_id = maybe_session_id

        payload_type = payload.get("type")
        if payload_type == "assistant":
            message = payload.get("message", {})
            content = message.get("content", [])
            text_blocks = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "\n".join(text for text in text_blocks if text)
            if joined:
                last_text_message = joined
            usage_payload = message.get("usage")
            if isinstance(usage_payload, dict):
                usage = _usage_from_payload(usage_payload)
            continue

        if payload_type != "result":
            continue

        usage_payload = payload.get("usage")
        if isinstance(usage_payload, dict):
            usage = _usage_from_payload(usage_payload)
        if payload.get("is_error") is True:
            error_text = payload.get("error") or payload.get("result") or "unknown error"
            raise CodexExecError(f"Local agent exec failed: {error_text}")
        maybe_result = payload.get("result")
        if isinstance(maybe_result, str) and maybe_result.strip():
            result_text = maybe_result.strip()

    if result_text:
        return result_text, usage, session_id
    if last_text_message:
        return last_text_message, usage, session_id
    raise CodexExecError("Local agent exec completed without a final message")


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
        raise CodexExecError(
            f"Local agent output did not match schema for {output_type.__name__}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CodexExecError(
            f"Local agent output was not valid JSON for {output_type.__name__}"
        ) from exc


def _safe_unlink(path: str | None) -> None:
    if path is None:
        return
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


def _augment_prompt_for_json(prompt: str, output_type: type[BaseModel]) -> str:
    schema = json.dumps(
        _normalize_json_schema(output_type.model_json_schema()),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return (
        f"{prompt}\n\n"
        "Return only minified JSON matching this schema. Do not include markdown fences.\n"
        f"{schema}"
    )


def _usage_from_payload(payload: dict[str, object]) -> CodexUsage:
    return CodexUsage(
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cached_input_tokens=int(
            payload.get("cached_input_tokens", payload.get("cache_read_input_tokens", 0)) or 0
        ),
        requests=1,
    )


def _write_json_schema(output_type: type[BaseModel], *, prefix: str) -> str:
    with tempfile.NamedTemporaryFile(
        prefix=prefix,
        suffix=".json",
        delete=False,
    ) as schema_file:
        schema_path = schema_file.name
        schema = _normalize_json_schema(output_type.model_json_schema())
        schema_file.write(json.dumps(schema, ensure_ascii=True).encode("utf-8"))
    return schema_path


def _render_custom_command(
    template: str,
    *,
    prompt: str,
    prompt_path: str,
    output_path: str,
    schema_path: str | None,
    model_name: str | None,
    cwd: str,
) -> str:
    values = {
        "prompt": shlex.quote(prompt),
        "prompt_file": shlex.quote(prompt_path),
        "output_path": shlex.quote(output_path),
        "output_json_path": shlex.quote(output_path),
        "schema_path": shlex.quote(schema_path or ""),
        "model": shlex.quote(model_name or settings.default_model),
        "cwd": shlex.quote(cwd),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise CodexExecError(
            f"Unknown placeholder in agent_exec_command_template: {exc.args[0]}"
        ) from exc


def _extract_command_executable(command_template: str) -> str | None:
    try:
        tokens = shlex.split(command_template)
    except ValueError:
        return None
    if not tokens:
        return None
    first_token = tokens[0]
    if "{" in first_token or "}" in first_token:
        return None
    return first_token


def _normalize_harness_name(executable: str) -> str:
    command_name = Path(executable).name.strip().lower()
    return command_name.removesuffix(".exe")


def _field_is_overridden(settings_obj: Settings, field_name: str) -> bool:
    field = Settings.model_fields[field_name]
    default_value = field.default_factory() if field.default_factory is not None else field.default
    return getattr(settings_obj, field_name) != default_value
