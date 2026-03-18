import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from app import cli
from app.models.review import FollowupMemory, FollowupSourceCard, UrlRecord
from app.services import followup
from app.services.followup import (
    answer_followup_question,
    build_followup_answer_prompt,
    load_followup_memory,
    rank_followup_source_cards,
)
from app.services.storage import init_db, insert_urls

runner = CliRunner()


def test_build_followup_answer_prompt_includes_context() -> None:
    memory = FollowupMemory(
        run_id="run-123",
        prompt="Best office chair",
        synthesis_markdown="Saved synthesis text",
        source_cards=[
            FollowupSourceCard(
                lane_name="Reviews",
                lane_goal="Find owner feedback",
                url="https://example.com/review",
                title="Review",
                source_query="office chair review",
                source_kind="web",
                distilled_text="### Highlights\n- Comfortable for long sessions.",
                relevance_score=80,
            )
        ],
    )

    prompt = build_followup_answer_prompt(memory, "How is lumbar support?", memory.source_cards)

    assert "Saved synthesis text" in prompt
    assert "How is lumbar support?" in prompt
    assert "https://example.com/review" in prompt


def test_rank_followup_source_cards_prefers_question_overlap() -> None:
    memory = FollowupMemory(
        run_id="run-123",
        prompt="Best office chair",
        synthesis_markdown="Summary",
        source_cards=[
            FollowupSourceCard(
                lane_name="Comfort",
                lane_goal="Find comfort feedback",
                url="https://example.com/lumbar",
                title="Lumbar Review",
                source_query="lumbar support review",
                source_kind="web",
                distilled_text="### Highlights\n- Strong lumbar support with adjustable depth.",
                relevance_score=65,
            ),
            FollowupSourceCard(
                lane_name="Assembly",
                lane_goal="Find setup notes",
                url="https://example.com/assembly",
                title="Assembly Notes",
                source_query="assembly review",
                source_kind="web",
                distilled_text="### Highlights\n- Took 20 minutes to assemble.",
                relevance_score=90,
            ),
        ],
    )

    ranked = rank_followup_source_cards(memory, "How good is the lumbar support?")

    assert ranked[0].url == "https://example.com/lumbar"


def test_load_followup_memory_rebuilds_from_storage(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "reviewbuddy.db"
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()
    markdown_path = run_dir / "source.md"
    markdown_path.write_text(
        "# Source\nURL: https://example.com/review\n\nThe chair has very good lumbar support.",
        encoding="utf-8",
    )
    transcript_path = run_dir / "youtube_transcripts.json"
    transcript_path.write_text(
        (
            '[{"url":"https://youtube.com/watch?v=abc","title":"Video Review",'
            '"transcript":"The lumbar support is adjustable and firm."}]'
        ),
        encoding="utf-8",
    )

    async def _run() -> FollowupMemory:
        await init_db(db_path)
        await insert_urls(
            db_path,
            [
                UrlRecord(
                    run_id="run-123",
                    url="https://example.com/review",
                    title="Review",
                    source_query="Comfort: lumbar support review",
                    status="fetched",
                    markdown_path=markdown_path,
                )
            ],
        )
        return await load_followup_memory(
            run_id="run-123",
            run_dir=run_dir,
            prompt="Best office chair",
            synthesis_markdown="Saved synthesis",
        )

    monkeypatch.setattr(followup.settings, "database_path", db_path)
    memory = asyncio.run(_run())

    assert len(memory.source_cards) == 2
    assert any(card.url == "https://example.com/review" for card in memory.source_cards)
    assert any(card.url == "https://youtube.com/watch?v=abc" for card in memory.source_cards)


@pytest.mark.asyncio
async def test_answer_followup_question_uses_saved_memory(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_run_codex_prompt(prompt: str, **_kwargs):  # noqa: ANN001
        captured["prompt"] = prompt
        return SimpleNamespace(message="Answer text")

    monkeypatch.setattr("app.services.followup.run_codex_prompt", fake_run_codex_prompt)

    memory = FollowupMemory(
        run_id="run-123",
        prompt="Best office chair",
        synthesis_markdown="Saved synthesis text",
        source_cards=[
            FollowupSourceCard(
                lane_name="Comfort",
                lane_goal="Find comfort feedback",
                url="https://example.com/review",
                title="Review",
                source_query="lumbar support review",
                source_kind="web",
                distilled_text="### Highlights\n- Strong lumbar support with adjustable depth.",
                relevance_score=80,
            )
        ],
    )

    answer = await answer_followup_question(memory, "How is lumbar support?")

    assert answer == "Answer text"
    assert "How is lumbar support?" in captured["prompt"]
    assert "Saved synthesis text" in captured["prompt"]
    assert "https://example.com/review" in captured["prompt"]


def test_ask_command_answers_previous_session(monkeypatch, tmp_path: Path) -> None:
    report = cli.ReviewRunResult(
        run_id="run-123",
        prompt="Best office chair",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        stats=cli.ReviewRunStats(total_urls=4, fetched=4, failed=0),
        synthesis_markdown="Saved synthesis text",
    )
    state = cli.FollowupSessionState(
        run_id="run-123",
        run_dir=tmp_path / "run-123",
        prompt="Best office chair",
        synthesis_markdown="Saved synthesis text",
    )
    memory = FollowupMemory(
        run_id="run-123",
        prompt="Best office chair",
        synthesis_markdown="Saved synthesis text",
        source_cards=[],
    )

    async def fake_load_state(run_id: str, output_dir: Path | None = None):  # noqa: ANN202
        assert run_id == "run-123"
        assert output_dir is None
        return report, state

    async def fake_ensure_memory(_state):  # noqa: ANN001, ANN202
        return memory

    async def fake_answer(memory_arg, question: str, model_name: str | None = None):  # noqa: ANN001, ANN202
        assert memory_arg is memory
        assert question == "How is lumbar support?"
        assert model_name is None
        return "Answer text"

    monkeypatch.setattr(cli, "_load_followup_state_for_run", fake_load_state)
    monkeypatch.setattr(cli, "_ensure_followup_memory", fake_ensure_memory)
    monkeypatch.setattr(cli, "answer_followup_question", fake_answer)

    result = runner.invoke(cli.app, ["ask", "run-123", "How is lumbar support?"])

    assert result.exit_code == 0
    assert "Answer text" in result.stdout
