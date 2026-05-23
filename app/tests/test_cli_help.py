import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from app import cli
from app.cli_doctor import DoctorCheck
from app.cli_help import build_command_reference
from app.models.review import RunRecord
from app.services.setup_runtime import SetupAction, SetupResult

runner = CliRunner()


async def _noop_init_db(_db_path) -> None:  # noqa: ANN001
    return None


def _run_record(run_id: str = "run-123", status: str = "completed") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        prompt="Best office chair for long coding sessions with strong lumbar support",
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        status=status,
        max_urls=10,
        max_agents=4,
        headful=True,
        output_dir=Path("/tmp") / run_id,
    )


def test_build_command_reference_includes_agent_first_commands() -> None:
    reference = build_command_reference()

    assert 'researchbuddy start "<prompt>"' in reference
    assert 'researchbuddy followup "<question>"' in reference
    assert "researchbuddy status [--run-id RUN_ID] [--json]" in reference
    assert "researchbuddy watch [--run-id RUN_ID]" in reference
    assert "researchbuddy doctor [--fix]" in reference
    assert "researchbuddy skills install openclaw" in reference
    assert "researchbuddy list [--limit 20] [--json]" in reference
    assert "researchbuddy run" not in reference
    assert "researchbuddy runs" not in reference


def test_build_command_reference_agent_mode_is_machine_friendly() -> None:
    reference = build_command_reference(agent=True)

    assert "ResearchBuddy CLI For Agents" in reference
    assert "## start" in reference
    assert "## status" in reference
    assert "## skills install" in reference
    assert "Purpose:" in reference
    assert 'Usage: `researchbuddy followup "<question>" [--run-id RUN_ID]' in reference


def test_top_level_help_uses_agent_first_surface() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "Start one ResearchBuddy prompt at a time" in result.stdout
    assert "start" in result.stdout
    assert "followup" in result.stdout
    assert "status" in result.stdout
    assert "watch" in result.stdout
    assert "doctor" in result.stdout
    assert "skills" in result.stdout
    assert "list" in result.stdout
    assert "runs" not in result.stdout


def test_top_level_invocation_defaults_to_help() -> None:
    result = runner.invoke(cli.app, [])

    assert result.exit_code == 0
    assert "Usage: researchbuddy [OPTIONS] COMMAND [ARGS]..." in result.stdout
    assert "start" in result.stdout
    assert "doctor" in result.stdout


def test_skills_install_openclaw_workspace_json(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "research"
    source_path.mkdir(parents=True)
    (source_path / "SKILL.md").write_text(
        "---\nname: research\ndescription: Test skill.\n---\n",
        encoding="utf-8",
    )
    workspace_path = tmp_path / "workspace"

    result = runner.invoke(
        cli.app,
        [
            "skills",
            "install",
            "openclaw",
            "--scope",
            "workspace",
            "--workspace",
            str(workspace_path),
            "--source",
            str(source_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "installed"
    assert payload["target_path"] == str(workspace_path / "skills" / "research")
    assert not (workspace_path / "skills" / "research").is_symlink()
    assert (workspace_path / "skills" / "research" / "SKILL.md").is_file()


def test_old_run_command_is_not_registered() -> None:
    result = runner.invoke(cli.app, ["run", "Best office chair"])

    assert result.exit_code == 2
    assert "No such command 'run'." in result.output
    assert "researchbuddy commands" not in result.output


def test_list_command_lists_recent_runs_and_current_marker(monkeypatch) -> None:
    async def fake_list_runs(_db_path, limit=20):  # noqa: ANN001, ANN202
        assert limit == 5
        return [_run_record()]

    async def fake_current(_db_path):  # noqa: ANN001, ANN202
        return "run-123"

    monkeypatch.setattr(cli, "list_runs", fake_list_runs)
    monkeypatch.setattr(cli, "fetch_current_run_id", fake_current)
    monkeypatch.setattr(cli, "init_db", _noop_init_db)

    result = runner.invoke(cli.app, ["list", "--limit", "5"])

    assert result.exit_code == 0
    assert "# Runs" in result.stdout
    assert "* run-123" in result.stdout
    assert "completed" in result.stdout


def test_list_command_prints_json(monkeypatch) -> None:
    async def fake_list_runs(_db_path, limit=20):  # noqa: ANN001, ANN202
        del limit
        return [_run_record()]

    async def fake_current(_db_path):  # noqa: ANN001, ANN202
        return "run-123"

    monkeypatch.setattr(cli, "list_runs", fake_list_runs)
    monkeypatch.setattr(cli, "fetch_current_run_id", fake_current)
    monkeypatch.setattr(cli, "init_db", _noop_init_db)

    result = runner.invoke(cli.app, ["list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["run_id"] == "run-123"
    assert payload[0]["is_current"] is True


def test_start_command_creates_current_run_and_spawns_worker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda _level: None)
    monkeypatch.setattr(cli.settings, "database_path", tmp_path / "test.db")
    monkeypatch.setattr(cli.settings, "storage_path", tmp_path / "storage")
    monkeypatch.setattr(cli, "_spawn_worker", lambda run_id, request_file, log_path: 4242)

    result = runner.invoke(cli.app, ["start", "Best office chair"])

    assert result.exit_code == 0
    assert "ResearchBuddy started" in result.stdout
    assert "PID: 4242" in result.stdout
    assert "researchbuddy status --run-id" in result.stdout


def test_start_command_prints_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda _level: None)
    monkeypatch.setattr(cli.settings, "database_path", tmp_path / "test.db")
    monkeypatch.setattr(cli.settings, "storage_path", tmp_path / "storage")
    monkeypatch.setattr(cli, "_spawn_worker", lambda run_id, request_file, log_path: 4242)

    result = runner.invoke(cli.app, ["start", "Best office chair", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["pid"] == 4242
    assert payload["prompt"] == "Best office chair"
    assert payload["status"] == "in_progress"
    assert payload["status_command"] == f"researchbuddy status --run-id {payload['run_id']}"
    assert payload["watch_command"] == f"researchbuddy watch --run-id {payload['run_id']}"


def test_start_command_refuses_live_run(monkeypatch) -> None:
    async def fake_live_run():  # noqa: ANN202
        return "run-123", 4242

    monkeypatch.setattr(cli, "_find_live_run", fake_live_run)

    result = runner.invoke(cli.app, ["start", "Best office chair"])

    assert result.exit_code == 1
    assert "Run already in progress: run-123" in result.stdout


def test_status_defaults_to_current_run(monkeypatch) -> None:
    async def fake_current(_db_path):  # noqa: ANN001, ANN202
        return "run-123"

    async def fake_status(run_id, output_dir=None):  # noqa: ANN001, ANN202
        assert run_id == "run-123"
        del output_dir
        return {
            "run_id": "run-123",
            "prompt": "Best office chair",
            "created_at": "2026-01-01T12:00:00+00:00",
            "status": "completed",
            "pid": 123,
            "is_process_running": False,
            "output_dir": "/tmp/run-123",
            "log_path": "/tmp/run-123/worker.log",
            "synthesis_path": "/tmp/run-123/synthesis.md",
            "urls": {"total": 4, "fetched": 3, "failed": 1},
            "started_at": None,
            "finished_at": None,
            "exit_code": 0,
        }

    monkeypatch.setattr(cli, "fetch_current_run_id", fake_current)
    monkeypatch.setattr(cli, "_run_status_payload", fake_status)

    result = runner.invoke(cli.app, ["status"])

    assert result.exit_code == 0
    assert "Run ID: run-123" in result.stdout
    assert "URLs: total=4 fetched=3 failed=1" in result.stdout


def test_status_prints_json(monkeypatch) -> None:
    async def fake_status(run_id, output_dir=None):  # noqa: ANN001, ANN202
        del output_dir
        return {
            "run_id": run_id,
            "prompt": "Best office chair",
            "created_at": "2026-01-01T12:00:00+00:00",
            "status": "completed",
            "pid": None,
            "is_process_running": False,
            "output_dir": "/tmp/run-123",
            "log_path": "/tmp/run-123/worker.log",
            "synthesis_path": None,
            "urls": {"total": 0, "fetched": 0, "failed": 0},
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
        }

    monkeypatch.setattr(cli, "_run_status_payload", fake_status)

    result = runner.invoke(cli.app, ["status", "--run-id", "run-123", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["run_id"] == "run-123"


def test_watch_exits_on_stale_running_status(monkeypatch) -> None:
    async def fake_status(run_id, output_dir=None):  # noqa: ANN001, ANN202
        del output_dir
        return {
            "run_id": run_id,
            "prompt": "Best office chair",
            "created_at": "2026-01-01T12:00:00+00:00",
            "status": "in_progress",
            "pid": 123,
            "is_process_running": False,
            "output_dir": "/tmp/run-123",
            "log_path": "/tmp/run-123/worker.log",
            "synthesis_path": None,
            "urls": {"total": 0, "fetched": 0, "failed": 0},
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
        }

    monkeypatch.setattr(cli, "_run_status_payload", fake_status)

    result = runner.invoke(cli.app, ["watch", "--run-id", "run-123", "--timeout", "1"])

    assert result.exit_code == 1
    assert "Run worker is no longer running while status is in_progress" in result.stdout


def test_watch_times_out(monkeypatch) -> None:
    async def fake_status(run_id, output_dir=None):  # noqa: ANN001, ANN202
        del output_dir
        return {
            "run_id": run_id,
            "prompt": "Best office chair",
            "created_at": "2026-01-01T12:00:00+00:00",
            "status": "in_progress",
            "pid": None,
            "is_process_running": False,
            "output_dir": "/tmp/run-123",
            "log_path": "/tmp/run-123/worker.log",
            "synthesis_path": None,
            "urls": {"total": 0, "fetched": 0, "failed": 0},
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
        }

    monkeypatch.setattr(cli, "_run_status_payload", fake_status)

    result = runner.invoke(
        cli.app,
        ["watch", "--run-id", "run-123", "--timeout", "1", "--interval", "2"],
    )

    assert result.exit_code == 124
    assert "Timed out after 1s while watching run." in result.stdout


def test_followup_defaults_to_current_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda _level: None)

    async def fake_current(_db_path):  # noqa: ANN001, ANN202
        return "run-123"

    async def fake_load_state(run_id, output_dir=None):  # noqa: ANN001, ANN202
        del output_dir
        report = cli.ReviewRunResult(
            run_id=run_id,
            prompt="Best office chair",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            stats=cli.ReviewRunStats(total_urls=4, fetched=3, failed=1),
            synthesis_markdown="Saved synthesis text",
        )
        state = cli.FollowupSessionState(
            run_id=run_id,
            run_dir=tmp_path,
            prompt=report.prompt,
            synthesis_markdown=report.synthesis_markdown,
        )
        return report, state

    async def fake_ensure_memory(state):  # noqa: ANN001, ANN202
        return cli.FollowupMemory(
            run_id=state.run_id,
            prompt=state.prompt,
            synthesis_markdown=state.synthesis_markdown,
            source_cards=[],
        )

    async def fake_answer(memory, question, model_name=None):  # noqa: ANN001, ANN202
        del memory, model_name
        assert question == "What broke most often?"
        return "Answer text"

    monkeypatch.setattr(cli, "fetch_current_run_id", fake_current)
    monkeypatch.setattr(cli, "_load_followup_state_for_run", fake_load_state)
    monkeypatch.setattr(cli, "_ensure_followup_memory", fake_ensure_memory)
    monkeypatch.setattr(cli, "answer_followup_question", fake_answer)

    result = runner.invoke(cli.app, ["followup", "What broke most often?"])

    assert result.exit_code == 0
    assert "Loading run run-123 for follow-up." in result.stdout
    assert "Answer text" in result.stdout
    assert "Continue this conversation in ChatGPT:" in result.stdout


def test_doctor_command_returns_nonzero_on_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_doctor_checks",
        lambda _settings: [DoctorCheck(name="local agent harness", ok=False, detail="missing")],
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "[FAIL] local agent harness" in result.stdout


def test_doctor_command_fix_runs_setup_and_reports_results(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    def fake_run_setup(_settings, install_playwright):  # noqa: ANN001, ANN202
        captured["install_playwright"] = install_playwright
        return SetupResult(
            actions=[SetupAction(name="playwright browsers", ok=True, detail="installed")],
            doctor_checks=[
                DoctorCheck(name="playwright browsers", ok=True, detail="Chromium launch OK")
            ],
        )

    async def fake_in_progress(_db_path):  # noqa: ANN001, ANN202
        return []

    monkeypatch.setattr(cli, "run_setup", fake_run_setup)
    monkeypatch.setattr(cli, "list_in_progress_runs", fake_in_progress)
    monkeypatch.setattr(cli, "init_db", _noop_init_db)

    result = runner.invoke(cli.app, ["doctor", "--fix", "--skip-playwright"])

    assert result.exit_code == 0
    assert captured["install_playwright"] is False
    assert "ResearchBuddy Setup" in result.stdout
    assert "ResearchBuddy Doctor" in result.stdout
    assert "[OK] playwright browsers" in result.stdout


def test_worker_runs_precreated_request(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda _level: None)
    monkeypatch.setattr(cli.settings, "database_path", tmp_path / "test.db")
    request = cli.ReviewRunRequest(
        prompt="Best office chair",
        max_urls=4,
        max_agents=2,
        headful=True,
        navigation_timeout_ms=1000,
        output_dir=tmp_path / "storage",
        research_mode=None,
    )
    request_file = tmp_path / "request.json"
    request_file.write_text(request.model_dump_json(), encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_run_review(request_obj, _deps, reporter=None, run_id=None, create_record=True):
        del reporter
        captured["prompt"] = request_obj.prompt
        captured["run_id"] = run_id
        captured["create_record"] = create_record
        return cli.ReviewRunResult(
            run_id=run_id,
            prompt=request_obj.prompt,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            stats=cli.ReviewRunStats(total_urls=0, fetched=0, failed=0),
            synthesis_markdown="Saved synthesis text",
        )

    monkeypatch.setattr(cli, "run_review", fake_run_review)

    result = runner.invoke(
        cli.app,
        ["_worker", "run-123", "--request-file", str(request_file)],
    )

    assert result.exit_code == 0
    assert captured == {
        "prompt": "Best office chair",
        "run_id": "run-123",
        "create_record": False,
    }
