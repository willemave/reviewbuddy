"""RLM-style REPL execution engine."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from app.agents.base import LaneRefinement
from app.agents.rlm_agents import rlm_root_agent, rlm_subquery_agent
from app.core.settings import get_settings
from app.models.review import LaneResult
from app.models.rlm import ContextDocument, RlmRefineRequest, RlmRunRequest, RlmRunResult
from app.services.storage import url_to_filename
from app.services.usage_tracker import UsageTracker
from app.services.youtube_transcriber import YouTubeTranscript

logger = logging.getLogger(__name__)
settings = get_settings()

FINAL_VAR_RE = re.compile(r"FINAL_VAR\((?P<var>[A-Za-z_][A-Za-z0-9_]*)\)")
FINAL_RE = re.compile(r"FINAL\((?P<content>.*)\)", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```repl\s*(?P<code>.*?)```", re.DOTALL)


async def run_rlm(request: RlmRunRequest) -> RlmRunResult:
    """Run an RLM-style interactive loop.

    Args:
        request: RlmRunRequest payload.

    Returns:
        RlmRunResult with final output.
    """

    max_iterations = request.max_iterations or settings.rlm_max_iterations
    root_model = request.root_model or settings.rlm_root_model
    subquery_model = request.subquery_model or settings.rlm_subquery_model

    logger.info(
        "RLM run starting",
        extra={
            "run_id": request.run_id,
            "docs": len(request.context_docs),
            "root_model": root_model,
            "subquery_model": subquery_model,
            "max_iterations": max_iterations,
        },
    )

    context_payload = [doc.model_dump() for doc in request.context_docs]
    repl_globals = _build_repl_globals(
        context_payload,
        request.deps,
        request.usage_tracker,
        subquery_model,
    )

    history: list[str] = []
    for iteration in range(1, max_iterations + 1):
        prompt = _build_root_prompt(request.prompt, history, context_payload)
        logger.info(
            "RLM iteration",
            extra={"run_id": request.run_id, "iteration": iteration},
        )
        result = await rlm_root_agent.run(
            prompt,
            deps=request.deps,
            model=root_model,
        )
        output_text = _coerce_text(result.output)
        if request.usage_tracker is not None:
            await request.usage_tracker.add(result.usage(), model_name=root_model)

        final = _extract_final(output_text, repl_globals)
        if final is not None:
            logger.info(
                "RLM run completed",
                extra={"run_id": request.run_id, "iteration": iteration},
            )
            return RlmRunResult(
                output=final,
                iterations=iteration,
                completed=True,
                completed_at=datetime.now(UTC),
            )

        repl_outputs = _execute_repl_blocks(output_text, repl_globals)
        history.append(f"ASSISTANT:\n{output_text}")
        if repl_outputs:
            history.append(f"REPL OUTPUT:\n{repl_outputs}")

    logger.warning(
        "RLM run reached max iterations",
        extra={"run_id": request.run_id, "iterations": max_iterations},
    )
    return RlmRunResult(
        output=output_text,
        iterations=max_iterations,
        completed=False,
        completed_at=datetime.now(UTC),
    )


async def refine_lane_queries_rlm(request: RlmRefineRequest) -> LaneRefinement:
    """Generate lane refinement queries using RLM.

    Args:
        request: RlmRefineRequest payload.

    Returns:
        LaneRefinement parsed from FINAL output.
    """

    prompt = (
        "You are refining search queries for a research lane. Use the REPL context to find "
        "gaps, notable brands, or user terminology. Produce JSON with shape: "
        '{"queries":[{"query":...,"rationale":...}]} containing 1-3 items. '
        "Do not include commentary outside JSON."
        f"\n\nUser prompt: {request.prompt}\nLane: {request.lane_name}\nGoal: {request.lane_goal}"
    )

    result = await run_rlm(
        RlmRunRequest(
            run_id=request.run_id,
            prompt=prompt,
            context_docs=request.context_docs,
            deps=request.deps,
            usage_tracker=request.usage_tracker,
            root_model=request.root_model,
            subquery_model=request.subquery_model,
            max_iterations=request.max_iterations,
        )
    )

    parsed = _parse_json_output(result.output)
    return LaneRefinement.model_validate(parsed)


def build_context_documents(
    lane_results: list[LaneResult],
    markdown_dir: Path,
    youtube_transcripts: list[YouTubeTranscript],
) -> list[ContextDocument]:
    """Build context documents for REPL use.

    Args:
        lane_results: Lane results with URLs.
        markdown_dir: Directory of markdown files.
        youtube_transcripts: YouTube transcripts.

    Returns:
        List of ContextDocument entries.
    """

    documents: list[ContextDocument] = []
    for lane in lane_results:
        for task in lane.url_tasks:
            path = markdown_dir / url_to_filename(task.url, ".md")
            content = ""
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(
                ContextDocument(
                    lane_name=lane.lane_name,
                    lane_goal=lane.goal,
                    url=task.url,
                    title=task.title,
                    kind="web",
                    content=content,
                    char_len=len(content),
                )
            )

    for transcript in youtube_transcripts:
        content = transcript.transcript
        documents.append(
            ContextDocument(
                lane_name=None,
                lane_goal=None,
                url=transcript.url,
                title=transcript.title,
                kind="youtube",
                content=content,
                char_len=len(content),
            )
        )

    return documents


def _build_root_prompt(prompt: str, history: list[str], context_payload: list[dict]) -> str:
    metadata = {
        "documents": len(context_payload),
        "total_chars": sum(doc.get("char_len", 0) for doc in context_payload),
    }
    history_block = "\n\n".join(history[-8:]) if history else ""
    return (
        "You can access the REPL environment (variable `context`) to answer the user prompt. "
        "Use ```repl``` code blocks to inspect context and run searches. Avoid unnecessary "
        "subqueries.\n\n"
        f"Context metadata: {json.dumps(metadata, ensure_ascii=True)}\n\n"
        f"User prompt: {prompt}\n\n"
        + (f"History:\n{history_block}\n\n" if history_block else "")
        + "Your next action:"
    )


def _build_repl_globals(
    context_payload: list[dict],
    deps,
    usage_tracker: UsageTracker | None,
    subquery_model: str,
) -> dict:
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "enumerate": enumerate,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "any": any,
        "all": all,
        "zip": zip,
    }

    def _record_usage(result, model_name: str) -> None:
        if usage_tracker is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(usage_tracker.add(result.usage(), model_name=model_name))
        except RuntimeError:
            return

    def llm_query(prompt: str, model: str | None = None) -> str:
        result = rlm_subquery_agent.run_sync(
            prompt,
            deps=deps,
            model=model or subquery_model,
        )
        _record_usage(result, model or subquery_model)
        return _coerce_text(result.output)

    def llm_query_batched(prompts: list[str], model: str | None = None) -> list[str]:
        return [llm_query(p, model=model) for p in prompts]

    return {
        "__builtins__": safe_builtins,
        "context": context_payload,
        "llm_query": llm_query,
        "llm_query_batched": llm_query_batched,
        "json": json,
        "re": re,
    }


def _execute_repl_blocks(response: str, repl_globals: dict) -> str:
    outputs: list[str] = []
    for match in CODE_BLOCK_RE.finditer(response):
        code = match.group("code").strip()
        if not code:
            continue
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, repl_globals)
        except Exception as exc:  # noqa: BLE001
            outputs.append(f"ERROR: {exc}")
        out_text = stdout.getvalue().strip()
        err_text = stderr.getvalue().strip()
        if out_text:
            outputs.append(out_text)
        if err_text:
            outputs.append(err_text)
    return "\n".join(outputs).strip()


def _extract_final(response: str, repl_globals: dict) -> str | None:
    var_match = FINAL_VAR_RE.search(response)
    if var_match:
        value = repl_globals.get(var_match.group("var"))
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=True)

    final_match = FINAL_RE.search(response)
    if final_match:
        return final_match.group("content").strip()
    return None


def _parse_json_output(value: str) -> dict:
    text = value.strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def _coerce_text(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=True)
