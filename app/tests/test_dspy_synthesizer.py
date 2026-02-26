import json

import pytest

from app.models.rlm import ContextDocument
from app.services.dspy_synthesizer import (
    _build_source_tree,
    _ensure_deno_available,
    _preview_text,
    _write_debug_payload,
)


def test_build_source_tree_dedupes_and_truncates() -> None:
    docs = [
        ContextDocument(
            lane_name="Lane A",
            lane_goal="Goal A",
            url="https://example.com/a",
            title="Doc A",
            kind="web",
            content="a" * 10,
            char_len=10,
        ),
        ContextDocument(
            lane_name="Lane B",
            lane_goal="Goal B",
            url="https://example.com/a",
            title="Doc B",
            kind="web",
            content="b" * 10,
            char_len=10,
        ),
    ]

    tree = _build_source_tree(docs, max_chars=5)

    keys = list(tree.keys())
    assert len(keys) == 2
    assert keys[0] == "https://example.com/a"
    assert keys[1].startswith("https://example.com/a#")
    assert tree[keys[0]].endswith("a" * 5)
    assert tree[keys[1]].endswith("b" * 5)


def test_ensure_deno_available_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr("app.services.dspy_synthesizer.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError):
        _ensure_deno_available()


def test_preview_text_truncates_and_squashes_whitespace() -> None:
    text = "Line one\n\nLine two   with  spaces"
    assert _preview_text(text, limit=100) == "Line one Line two with spaces"
    assert _preview_text(text, limit=10) == "Line one L..."


def test_write_debug_payload(tmp_path) -> None:
    payload = {"answer": "ok"}
    _write_debug_payload(tmp_path, payload)
    debug_path = tmp_path / "dspy_rlm_debug.json"
    assert debug_path.exists()
    saved = json.loads(debug_path.read_text(encoding="utf-8"))
    assert saved["answer"] == "ok"
