"""DSPy RLM-based synthesizer for review reports."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

import dspy
from pydantic import ValidationError

from app.agents.base import ReviewSynthesis
from app.core.settings import get_settings
from app.models.rlm import ContextDocument, DspySynthesisRequest

logger = logging.getLogger(__name__)
settings = get_settings()
DEBUG_FILENAME = "dspy_rlm_debug.json"


class ReviewSynthesisRlm(dspy.Signature):
    """RLM signature for review synthesis."""

    directory: dict = dspy.InputField(
        desc=(
            "Mapping of source identifiers (usually URLs) to their content and metadata. "
            "Each value includes title, lane, and body text."
        )
    )
    query: str = dspy.InputField(desc="The user prompt to answer.")
    answer: str = dspy.OutputField(
        desc=(
            "Return JSON with keys: summary (string), key_findings (list of strings), "
            "recommendation (string), sources (list of {url,title,notes}), and gaps (list "
            "of strings). Use only source ids/URLs present in the directory keys."
        )
    )


async def synthesize_review_dspy(request: DspySynthesisRequest) -> ReviewSynthesis:
    """Synthesize review findings using DSPy RLM.

    Args:
        request: DSPy synthesis request with prompt and context docs.

    Returns:
        ReviewSynthesis output.
    """

    return await asyncio.to_thread(_run_dspy_rlm, request)


def _run_dspy_rlm(request: DspySynthesisRequest) -> ReviewSynthesis:
    _configure_dspy()
    deno_path = _ensure_deno_available()
    total_chars = sum(len(doc.content) for doc in request.context_docs)
    max_chars = settings.dspy_rlm_source_max_chars
    truncated = sum(
        1
        for doc in request.context_docs
        if max_chars > 0 and len(doc.content) > max_chars
    )
    source_tree = _build_source_tree(request.context_docs, settings.dspy_rlm_source_max_chars)
    logger.info(
        "DSPy RLM synthesis starting",
        extra={
            "model": settings.dspy_rlm_model,
            "max_iterations": request.max_iterations or settings.dspy_rlm_max_iterations,
            "max_llm_calls": request.max_llm_calls or settings.dspy_rlm_max_llm_calls,
            "sources": len(source_tree),
            "deno_path": deno_path,
            "total_chars": total_chars,
            "truncated_sources": truncated,
        },
    )
    rlm = dspy.RLM(
        ReviewSynthesisRlm,
        max_iterations=request.max_iterations or settings.dspy_rlm_max_iterations,
        max_llm_calls=request.max_llm_calls or settings.dspy_rlm_max_llm_calls,
        verbose=settings.dspy_rlm_verbose,
    )
    result = rlm(directory=source_tree, query=request.prompt)
    answer_text = str(result.answer)
    trajectory = getattr(result, "trajectory", None) or []
    if request.run_dir and settings.dspy_rlm_verbose:
        _write_debug_payload(
            request.run_dir,
            _build_debug_payload(
                request=request,
                result=result,
                deno_path=deno_path,
                total_chars=total_chars,
                truncated=truncated,
                sources=len(source_tree),
            ),
        )
    logger.info(
        "DSPy RLM synthesis completed",
        extra={
            "answer_chars": len(answer_text),
            "trajectory_len": len(trajectory),
            "final_reasoning": getattr(result, "final_reasoning", None),
        },
    )
    if settings.dspy_rlm_verbose:
        logger.info("DSPy RLM answer preview: %s", _preview_text(answer_text, limit=400))
    return _parse_synthesis(result.answer)


def _configure_dspy() -> None:
    if not settings.dspy_rlm_api_key:
        raise ValueError(
            "DSPy RLM summarizer requires DSPY_RLM_API_KEY or CEREBRAS_API_KEY to be set."
        )
    lm = dspy.LM(
        settings.dspy_rlm_model,
        api_key=settings.dspy_rlm_api_key,
        api_base=settings.dspy_rlm_api_base,
        temperature=settings.dspy_rlm_temperature,
        top_p=settings.dspy_rlm_top_p,
        max_tokens=settings.dspy_rlm_max_tokens,
        disable_reasoning=settings.dspy_rlm_disable_reasoning,
        clear_thinking=settings.dspy_rlm_clear_thinking,
    )
    dspy.settings.configure(lm=lm)


def _ensure_deno_available() -> str:
    deno_path = shutil.which("deno")
    if deno_path:
        return deno_path
    raise RuntimeError(
        "DSPy RLM requires Deno for its Python REPL sandbox. "
        "Install with `brew install deno` or set DENO_DIR if already installed."
    )


def _build_source_tree(
    context_docs: list[ContextDocument],
    max_chars: int,
) -> dict[str, str]:
    tree: dict[str, str] = {}
    for idx, doc in enumerate(context_docs, start=1):
        key = doc.url or doc.title or f"doc-{idx}"
        key = _dedupe_key(key, tree)
        content = doc.content[:max_chars] if max_chars > 0 else doc.content
        lane_name = doc.lane_name or "unknown"
        lane_goal = doc.lane_goal or "unknown"
        title = doc.title or "untitled"
        tree[key] = (
            f"Title: {title}\n"
            f"URL: {doc.url or ''}\n"
            f"Lane: {lane_name}\n"
            f"Goal: {lane_goal}\n"
            f"Kind: {doc.kind}\n\n"
            f"{content}"
        )
    return tree


def _dedupe_key(key: str, existing: dict[str, str]) -> str:
    if key not in existing:
        return key
    suffix = 2
    while f"{key}#{suffix}" in existing:
        suffix += 1
    return f"{key}#{suffix}"


def _parse_synthesis(raw: str) -> ReviewSynthesis:
    try:
        payload = _parse_json_output(raw)
        return ReviewSynthesis.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("DSPy RLM synthesis parse failed: %s", exc)
        logger.warning("DSPy RLM raw answer preview: %s", _preview_text(raw, limit=400))
        summary = raw.strip()
        if len(summary) > 4000:
            summary = summary[:4000].rstrip()
        return ReviewSynthesis(
            summary=summary,
            key_findings=[],
            recommendation="",
            sources=[],
            gaps=["DSPy output was not valid JSON."],
        )


def _parse_json_output(value: str) -> dict:
    text = value.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def _preview_text(value: str, limit: int = 400) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _build_debug_payload(
    request: DspySynthesisRequest,
    result: object,
    deno_path: str,
    total_chars: int,
    truncated: int,
    sources: int,
) -> dict:
    return {
        "prompt": request.prompt,
        "model": settings.dspy_rlm_model,
        "deno_path": deno_path,
        "max_iterations": request.max_iterations or settings.dspy_rlm_max_iterations,
        "max_llm_calls": request.max_llm_calls or settings.dspy_rlm_max_llm_calls,
        "sources": sources,
        "total_chars": total_chars,
        "truncated_sources": truncated,
        "answer": str(getattr(result, "answer", "")),
        "final_reasoning": getattr(result, "final_reasoning", None),
        "trajectory": getattr(result, "trajectory", None),
    }


def _write_debug_payload(run_dir: Path, payload: dict) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_path = run_dir / DEBUG_FILENAME
        debug_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        logger.info("DSPy RLM debug saved: %s", debug_path)
    except Exception as exc:
        logger.warning("Failed to write DSPy RLM debug payload: %s", exc)
