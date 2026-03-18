import asyncio
from datetime import UTC, datetime
from pathlib import Path

from app.models.review import RunRecord
from app.services.storage import create_run, fetch_run, init_db, resolve_run_dir


def test_fetch_run_returns_record(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    run_id = "run-123"
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    run = RunRecord(
        run_id=run_id,
        prompt="Test prompt",
        created_at=created_at,
        status="completed",
        max_urls=5,
        max_agents=2,
        headful=True,
        output_dir=tmp_path,
    )

    async def _run() -> None:
        await init_db(db_path)
        await create_run(db_path, run)
        fetched = await fetch_run(db_path, run_id)
        assert fetched is not None
        assert fetched.run_id == run_id
        assert fetched.prompt == run.prompt
        assert fetched.output_dir == tmp_path

    asyncio.run(_run())


def test_fetch_run_returns_none(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"

    async def _run() -> None:
        await init_db(db_path)
        fetched = await fetch_run(db_path, "missing")
        assert fetched is None

    asyncio.run(_run())


def test_resolve_run_dir_uses_stored_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-123"
    assert resolve_run_dir(run_dir, "run-123") == run_dir


def test_resolve_run_dir_uses_override_base_dir(tmp_path: Path) -> None:
    override_dir = tmp_path / "storage"
    assert (
        resolve_run_dir(tmp_path / "old-run", "run-123", override_dir) == override_dir / "run-123"
    )


def test_resolve_run_dir_accepts_override_run_dir(tmp_path: Path) -> None:
    override_dir = tmp_path / "run-123"
    assert resolve_run_dir(tmp_path / "old-run", "run-123", override_dir) == override_dir
