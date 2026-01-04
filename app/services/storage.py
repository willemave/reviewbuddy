"""SQLite storage helpers for ReviewBuddy."""

import hashlib
from datetime import datetime
from pathlib import Path

import aiosqlite

from app.constants import (
    RUN_STATUS_IN_PROGRESS,
    URL_STATUS_FAILED,
    URL_STATUS_FETCHED,
    URL_STATUS_PENDING,
)
from app.models.review import RunRecord, UrlRecord


def build_run_paths(base_dir: Path, run_id: str) -> dict[str, Path]:
    """Create and return run-specific directories.

    Args:
        base_dir: Base storage directory.
        run_id: Run identifier.

    Returns:
        Mapping of directory names to paths.
    """

    run_dir = base_dir / run_id
    html_dir = run_dir / "html"
    markdown_dir = run_dir / "markdown"
    lanes_dir = run_dir / "lanes"
    videos_dir = run_dir / "videos"
    transcripts_dir = run_dir / "transcripts"
    pdf_dir = run_dir / "pdf"

    html_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    lanes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run": run_dir,
        "html": html_dir,
        "markdown": markdown_dir,
        "lanes": lanes_dir,
        "videos": videos_dir,
        "transcripts": transcripts_dir,
        "pdf": pdf_dir,
    }


def url_to_filename(url: str, suffix: str) -> str:
    """Create a stable filename for a URL.

    Args:
        url: The source URL.
        suffix: File extension suffix (e.g. ".html").

    Returns:
        Filename string.
    """

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{digest}{suffix}"


async def init_db(db_path: Path) -> None:
    """Initialize SQLite schema.

    Args:
        db_path: Path to SQLite database.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                max_urls INTEGER NOT NULL,
                max_agents INTEGER NOT NULL,
                headful INTEGER NOT NULL,
                output_dir TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                source_query TEXT NOT NULL,
                status TEXT NOT NULL,
                html_path TEXT,
                markdown_path TEXT,
                error TEXT,
                fetched_at TEXT,
                UNIQUE(run_id, url)
            );
            """
        )
        await conn.commit()


async def create_run(db_path: Path, run: RunRecord) -> None:
    """Insert a run record.

    Args:
        db_path: SQLite path.
        run: Run record.
    """

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            INSERT INTO runs (run_id, prompt, created_at, status, max_urls, max_agents, headful, output_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.prompt,
                run.created_at.isoformat(),
                run.status,
                run.max_urls,
                run.max_agents,
                1 if run.headful else 0,
                str(run.output_dir),
            ),
        )
        await conn.commit()


async def fetch_run(db_path: Path, run_id: str) -> RunRecord | None:
    """Fetch a run record by ID.

    Args:
        db_path: SQLite path.
        run_id: Run identifier.

    Returns:
        RunRecord if found, otherwise None.
    """

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            """
            SELECT run_id, prompt, created_at, status, max_urls, max_agents, headful, output_dir
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    created_at = datetime.fromisoformat(row[2])
    return RunRecord(
        run_id=row[0],
        prompt=row[1],
        created_at=created_at,
        status=row[3],
        max_urls=int(row[4]),
        max_agents=int(row[5]),
        headful=bool(row[6]),
        output_dir=Path(row[7]),
    )


async def list_runs(db_path: Path, limit: int = 20) -> list[RunRecord]:
    """List recent run records.

    Args:
        db_path: SQLite path.
        limit: Max number of runs to return.

    Returns:
        List of RunRecord entries ordered by created_at desc.
    """

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            """
            SELECT run_id, prompt, created_at, status, max_urls, max_agents, headful, output_dir
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    records: list[RunRecord] = []
    for row in rows:
        created_at = datetime.fromisoformat(row[2])
        records.append(
            RunRecord(
                run_id=row[0],
                prompt=row[1],
                created_at=created_at,
                status=row[3],
                max_urls=int(row[4]),
                max_agents=int(row[5]),
                headful=bool(row[6]),
                output_dir=Path(row[7]),
            )
        )

    return records


async def update_run_status(db_path: Path, run_id: str, status: str) -> None:
    """Update run status.

    Args:
        db_path: SQLite path.
        run_id: Run identifier.
        status: New status.
    """

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?",
            (status, run_id),
        )
        await conn.commit()


async def insert_urls(db_path: Path, records: list[UrlRecord]) -> None:
    """Insert URL records.

    Args:
        db_path: SQLite path.
        records: URL records.
    """

    async with aiosqlite.connect(db_path) as conn:
        await conn.executemany(
            """
            INSERT OR IGNORE INTO urls (run_id, url, title, source_query, status, html_path, markdown_path, error, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.run_id,
                    record.url,
                    record.title,
                    record.source_query,
                    record.status,
                    str(record.html_path) if record.html_path else None,
                    str(record.markdown_path) if record.markdown_path else None,
                    record.error,
                    None,
                )
                for record in records
            ],
        )
        await conn.commit()


async def mark_url_fetched(
    db_path: Path,
    run_id: str,
    url: str,
    html_path: Path,
    markdown_path: Path,
) -> None:
    """Mark URL as fetched.

    Args:
        db_path: SQLite path.
        run_id: Run identifier.
        url: URL string.
        html_path: Path to stored HTML.
        markdown_path: Path to stored markdown.
    """

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            UPDATE urls
            SET status = ?, html_path = ?, markdown_path = ?, fetched_at = ?
            WHERE run_id = ? AND url = ?
            """,
            (
                URL_STATUS_FETCHED,
                str(html_path),
                str(markdown_path),
                datetime.utcnow().isoformat(),
                run_id,
                url,
            ),
        )
        await conn.commit()


async def mark_url_failed(db_path: Path, run_id: str, url: str, error: str) -> None:
    """Mark URL as failed.

    Args:
        db_path: SQLite path.
        run_id: Run identifier.
        url: URL string.
        error: Error message.
    """

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            UPDATE urls
            SET status = ?, error = ?, fetched_at = ?
            WHERE run_id = ? AND url = ?
            """,
            (
                URL_STATUS_FAILED,
                error,
                datetime.utcnow().isoformat(),
                run_id,
                url,
            ),
        )
        await conn.commit()


async def fetch_run_stats(db_path: Path, run_id: str) -> tuple[int, int, int]:
    """Fetch URL stats for a run.

    Args:
        db_path: SQLite path.
        run_id: Run identifier.

    Returns:
        Tuple of (total, fetched, failed).
    """

    async with aiosqlite.connect(db_path) as conn:
        total = await _count(conn, run_id, None)
        fetched = await _count(conn, run_id, URL_STATUS_FETCHED)
        failed = await _count(conn, run_id, URL_STATUS_FAILED)
    return total, fetched, failed


async def _count(conn: aiosqlite.Connection, run_id: str, status: str | None) -> int:
    if status:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM urls WHERE run_id = ? AND status = ?",
            (run_id, status),
        )
    else:
        cursor = await conn.execute("SELECT COUNT(*) FROM urls WHERE run_id = ?", (run_id,))
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


def new_run_record(
    run_id: str,
    prompt: str,
    max_urls: int,
    max_agents: int,
    headful: bool,
    output_dir: Path,
) -> RunRecord:
    """Build a RunRecord with default status.

    Args:
        run_id: Run identifier.
        prompt: Prompt text.
        max_urls: Max URLs.
        max_agents: Max agents.
        headful: Headful mode.
        output_dir: Output directory.

    Returns:
        RunRecord instance.
    """

    return RunRecord(
        run_id=run_id,
        prompt=prompt,
        created_at=datetime.utcnow(),
        status=RUN_STATUS_IN_PROGRESS,
        max_urls=max_urls,
        max_agents=max_agents,
        headful=headful,
        output_dir=output_dir,
    )


def new_url_record(
    run_id: str,
    url: str,
    title: str | None,
    source_query: str,
) -> UrlRecord:
    """Build a UrlRecord in pending status.

    Args:
        run_id: Run identifier.
        url: URL string.
        title: Title.
        source_query: Query that produced the URL.

    Returns:
        UrlRecord instance.
    """

    return UrlRecord(
        run_id=run_id,
        url=url,
        title=title,
        source_query=source_query,
        status=URL_STATUS_PENDING,
    )
