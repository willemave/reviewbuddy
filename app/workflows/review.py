"""Review workflow orchestration."""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

from app.agents.base import AgentDeps, LaneSpec, LaneSynthesis
from app.agents.lane_planner import plan_lanes
from app.agents.lane_refiner import refine_lane_queries
from app.agents.synthesizer import (
    build_final_synthesis_prompt,
    build_lane_synthesis_prompt,
    build_merge_synthesis_prompt,
    synthesize_lane,
    synthesize_merge_node,
    synthesize_review,
)
from app.constants import (
    FOLLOWUP_MEMORY_FILENAME,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    YOUTUBE_TRANSCRIPTS_FILENAME,
)
from app.core.logging import add_run_file_handler
from app.core.settings import get_settings
from app.models.review import (
    FollowupMemory,
    FollowupSourceCard,
    LaneResult,
    ReviewRunRequest,
    ReviewRunResult,
    ReviewRunStats,
    UrlTask,
)
from app.services.markdown_converter import MarkdownError, html_file_to_markdown
from app.services.playwright_fetcher import FetchError, capture_html, should_retry_headful
from app.services.query_shaper import QueryShapeRequest, shape_query
from app.services.reporter import RunReporter
from app.services.search_provider import SearchProviderError, build_search_provider
from app.services.storage import (
    build_run_paths,
    create_run,
    fetch_run_stats,
    init_db,
    mark_url_failed,
    mark_url_fetched,
    new_run_record,
    new_url_record,
    update_run_status,
    url_to_filename,
)
from app.services.token_estimator import estimate_tokens
from app.services.transcript_summarizer import summarize_youtube_transcripts
from app.services.url_handlers import CustomContent, fetch_custom_content
from app.services.usage_tracker import UsageSnapshot, UsageTracker
from app.services.youtube_transcriber import (
    YouTubeError,
    YouTubeTranscript,
    transcribe_youtube_videos_with_timeout,
)

settings = get_settings()
search_provider = build_search_provider(settings)
logger = logging.getLogger(__name__)

NUMERIC_SIGNAL_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:%|x|ms|gb|tb|w|wh|mah|hz|fps|in|inch|hours?|mins?|minutes?|years?)?\b",
    re.IGNORECASE,
)
CAVEAT_MARKERS = (
    "however",
    "but",
    "downside",
    "issue",
    "problem",
    "complaint",
    "failure",
    "tradeoff",
    "caveat",
    "limitation",
    "concern",
)
SIGNAL_MARKERS = (
    "recommend",
    "tested",
    "comparison",
    "durability",
    "reliability",
    "quiet",
    "noise",
    "battery",
    "performance",
    "warranty",
    "return",
)
NOISE_MARKERS = (
    "cookie",
    "privacy policy",
    "terms of service",
    "accept all",
    "sign in",
    "subscribe",
    "javascript",
    "advertisement",
)


@dataclass(frozen=True)
class CandidateUrl:
    """Candidate URL returned by search prior to ranking."""

    url: str
    title: str | None
    source_query: str
    lane_name: str
    score: float
    domain: str
    title_key: str
    provider_name: str | None = None
    provider_markdown: str | None = None
    provider_html: str | None = None


@dataclass(frozen=True)
class SourceCard:
    """Dense, ranked source excerpt used for hierarchical synthesis."""

    lane_name: str
    lane_goal: str
    url: str
    title: str | None
    source_query: str | None
    source_kind: str
    distilled_text: str
    relevance_score: int

    @property
    def value_density(self) -> float:
        """Return compactness-aware relevance density."""

        return self.relevance_score / max(len(self.distilled_text), 200)


@dataclass(frozen=True)
class LaneContextPacket:
    """Packed source cards for a single lane."""

    lane_name: str
    lane_goal: str
    cards: list[SourceCard]


@dataclass(frozen=True)
class LaneSummaryPacket:
    """Lane synthesis plus its packed supporting cards."""

    lane_name: str
    lane_goal: str
    synthesis: LaneSynthesis
    cards: list[SourceCard]


def _emit_reporter(
    reporter: RunReporter | None,
    attr: str,
    *args,
) -> None:
    if reporter is None:
        return
    handler = getattr(reporter, attr, None)
    if handler is None:
        return
    handler(*args)


async def run_review(
    request: ReviewRunRequest,
    deps: AgentDeps,
    reporter: RunReporter | None = None,
) -> ReviewRunResult:
    """Run the full review research workflow.

    Args:
        request: Review run request.
        deps: Agent dependencies.
        reporter: Optional progress reporter callbacks.

    Returns:
        ReviewRunResult with synthesis markdown.
    """

    logger.info("Starting review run")
    await init_db(settings.database_path)

    run_id = uuid.uuid4().hex
    run_paths = build_run_paths(request.output_dir, run_id)
    usage_tracker = UsageTracker()
    run_record = new_run_record(
        run_id=run_id,
        prompt=request.prompt,
        max_urls=request.max_urls,
        max_agents=request.max_agents,
        headful=request.headful,
        output_dir=run_paths["run"],
    )
    await create_run(settings.database_path, run_record)
    add_run_file_handler(run_paths["run"] / "run.log", settings.log_level)
    logger.info(
        "Run initialized",
        extra={
            "run_id": run_id,
            "max_urls": request.max_urls,
            "max_agents": request.max_agents,
            "headful_fallback": request.headful,
        },
    )

    try:
        youtube_transcripts: list[YouTubeTranscript] = []
        if settings.youtube_max_videos > 0:
            youtube_transcripts = await _collect_youtube_transcripts(
                request.prompt,
                run_paths["videos"],
                run_paths["transcripts"],
                settings.youtube_max_videos,
                settings.whisper_model,
                usage_tracker=usage_tracker,
            )

        lane_plan = await plan_lanes(
            request.prompt,
            deps,
            usage_tracker=usage_tracker,
            model_name=request.planner_model,
        )
        lanes = _select_lanes(lane_plan.lanes, request.max_agents)
        lanes = _allocate_lane_budgets(lanes, request.max_urls)
        logger.info("Planned %d lanes", len(lanes))
        _emit_reporter(reporter, "on_lanes_planned", len(lanes))
        for lane in lanes:
            logger.info(
                "Lane planned: %s",
                lane.name,
                extra={"goal": lane.goal, "budget": lane.url_budget},
            )

        async with async_playwright() as playwright:
            logger.info("Launching browser (headless, headful_fallback=%s)", request.headful)
            browser = await playwright.chromium.launch(headless=True)
            headful_fallback = build_headful_fallback(playwright, enabled=request.headful)
            try:
                lane_tasks = [
                    _run_lane(
                        run_id=run_id,
                        lane=lane,
                        prompt=request.prompt,
                        browser=browser,
                        headful_fallback=headful_fallback,
                        run_paths=run_paths,
                        timeout_ms=request.navigation_timeout_ms,
                        deps=deps,
                        usage_tracker=usage_tracker,
                        model_name=request.sub_agent_model,
                        reporter=reporter,
                    )
                    for lane in lanes
                ]
                lane_results = await asyncio.gather(*lane_tasks)
            finally:
                await browser.close()
                await close_headful_fallback(headful_fallback)
                logger.info("Browser closed")

        synthesis_markdown, usage_snapshot = await _synthesize(
            request.prompt,
            run_paths["markdown"],
            lane_results,
            youtube_transcripts,
            deps,
            usage_tracker,
            model_name=request.sub_agent_model,
        )
        logger.info("Synthesis complete")

        await update_run_status(settings.database_path, run_id, RUN_STATUS_COMPLETED)

        total, fetched, failed = await fetch_run_stats(settings.database_path, run_id)
        stats = ReviewRunStats(total_urls=total, fetched=fetched, failed=failed)
        logger.info("Run stats: %d total, %d fetched, %d failed", total, fetched, failed)
        _log_usage_snapshot(usage_snapshot)

        synthesis_path = run_paths["run"] / "synthesis.md"
        synthesis_path.write_text(synthesis_markdown, encoding="utf-8")
        _persist_followup_artifacts(
            run_dir=run_paths["run"],
            run_id=run_id,
            prompt=request.prompt,
            synthesis_markdown=synthesis_markdown,
            lane_results=lane_results,
            markdown_dir=run_paths["markdown"],
            youtube_transcripts=youtube_transcripts,
        )

        return ReviewRunResult(
            run_id=run_id,
            prompt=request.prompt,
            created_at=run_record.created_at,
            stats=stats,
            synthesis_markdown=synthesis_markdown,
        )
    except Exception:
        logger.exception("Run failed")
        await update_run_status(settings.database_path, run_id, RUN_STATUS_FAILED)
        raise


def _select_lanes(lanes: list[LaneSpec], max_agents: int) -> list[LaneSpec]:
    if not lanes:
        return []
    return lanes[:max_agents]


def _allocate_lane_budgets(lanes: list[LaneSpec], max_urls: int) -> list[LaneSpec]:
    if not lanes:
        return []

    requested = sum(lane.url_budget or 0 for lane in lanes)
    budgets: list[int] = []

    if requested <= 0:
        base = max(1, max_urls // len(lanes))
        budgets = [base for _ in lanes]
        remainder = max_urls - sum(budgets)
        for idx in range(remainder):
            budgets[idx % len(budgets)] += 1
    else:
        if max_urls <= len(lanes):
            budgets = [1 for _ in lanes]
            return [
                lane.model_copy(update={"url_budget": budget})
                for lane, budget in zip(lanes, budgets, strict=True)
            ]
        scale = max_urls / requested
        budgets = [max(1, int((lane.url_budget or 1) * scale)) for lane in lanes]
        diff = max_urls - sum(budgets)
        idx = 0
        while diff != 0:
            if diff > 0:
                budgets[idx % len(budgets)] += 1
                diff -= 1
            else:
                if budgets[idx % len(budgets)] > 1:
                    budgets[idx % len(budgets)] -= 1
                    diff += 1
            idx += 1

    return [
        lane.model_copy(update={"url_budget": budget})
        for lane, budget in zip(lanes, budgets, strict=True)
    ]


async def _run_lane(
    run_id: str,
    lane: LaneSpec,
    prompt: str,
    browser,
    headful_fallback: dict,
    run_paths: dict[str, Path],
    timeout_ms: int,
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
    reporter: RunReporter | None = None,
) -> LaneResult:
    budget = lane.url_budget or 0
    if budget <= 0:
        return LaneResult(lane_name=lane.name, goal=lane.goal, url_tasks=[])

    seen_urls: set[str] = set()
    lane_slug = _slugify(lane.name)
    lane_deps = deps.model_copy(update={"job_id": f"{deps.job_id}-{lane_slug}"})
    seed_budget = _seed_budget(budget)

    logger.info(
        "Lane starting: %s",
        lane.name,
        extra={"goal": lane.goal, "budget": budget, "seed_budget": seed_budget},
    )
    initial_tasks = await _collect_urls_for_queries(
        lane,
        lane.seed_queries,
        seed_budget,
        seen_urls,
    )
    logger.info(
        "Lane %s seed search collected %d urls",
        lane.name,
        len(initial_tasks),
    )
    _emit_reporter(reporter, "on_urls_discovered", len(initial_tasks))

    if initial_tasks:
        await _store_url_records(run_id, initial_tasks)

    all_tasks = list(initial_tasks)

    initial_batch_size = _initial_feedback_size(seed_budget, len(initial_tasks))
    first_batch = initial_tasks[:initial_batch_size]
    remaining_tasks = initial_tasks[initial_batch_size:]

    context = await browser.new_context()
    try:
        await _crawl_tasks(
            run_id,
            first_batch,
            context,
            headful_fallback,
            run_paths,
            timeout_ms,
            usage_tracker,
            reporter,
        )

        evidence_tasks = list(first_batch)
        refinement_targets = _refinement_targets(budget)
        max_rounds = max(0, settings.refinement_rounds)
        for round_index, target in enumerate(refinement_targets[:max_rounds], start=1):
            budget_remaining = max(0, budget - len(all_tasks))
            if budget_remaining <= 0 or not evidence_tasks:
                break

            evidence = _build_evidence_snippets(evidence_tasks, run_paths["markdown"])
            if not evidence.strip():
                break

            refinement = await refine_lane_queries(
                prompt,
                lane.name,
                lane.goal,
                evidence,
                lane_deps,
                usage_tracker=usage_tracker,
                model_name=model_name,
            )
            logger.info(
                "Lane %s refinement round %d produced %d queries",
                lane.name,
                round_index,
                len(refinement.queries),
            )
            new_tasks = await _collect_urls_for_queries(
                lane,
                refinement.queries,
                budget_remaining,
                seen_urls,
            )
            if new_tasks:
                await _store_url_records(run_id, new_tasks)
                all_tasks.extend(new_tasks)
                remaining_tasks.extend(new_tasks)
                logger.info(
                    "Lane %s refinement round %d collected %d urls",
                    lane.name,
                    round_index,
                    len(new_tasks),
                )
                _emit_reporter(reporter, "on_urls_discovered", len(new_tasks))
            else:
                logger.info("Lane %s refinement round %d added no urls", lane.name, round_index)
                continue

            feedback_size = max(
                1,
                min(len(new_tasks), _initial_feedback_size(target, len(new_tasks))),
            )
            feedback_batch = new_tasks[:feedback_size]
            if feedback_batch:
                await _crawl_tasks(
                    run_id,
                    feedback_batch,
                    context,
                    headful_fallback,
                    run_paths,
                    timeout_ms,
                    usage_tracker,
                    reporter,
                )
                feedback_urls = {task.url for task in feedback_batch}
                remaining_tasks = [
                    task for task in remaining_tasks if task.url not in feedback_urls
                ]
                evidence_tasks = (evidence_tasks + feedback_batch)[-target:]

        await _crawl_tasks(
            run_id,
            remaining_tasks,
            context,
            headful_fallback,
            run_paths,
            timeout_ms,
            usage_tracker,
            reporter,
        )
    finally:
        await context.close()

    lane_markdown = _build_lane_markdown(lane, all_tasks, run_paths["markdown"])
    lane_path = run_paths["lanes"] / f"{lane_slug}.md"
    lane_path.write_text(lane_markdown, encoding="utf-8")
    logger.info(
        "Lane completed: %s",
        lane.name,
        extra={"total_urls": len(all_tasks)},
    )
    _emit_reporter(reporter, "on_lane_done", lane.name)

    return LaneResult(lane_name=lane.name, goal=lane.goal, url_tasks=all_tasks)


def _seed_budget(total_budget: int) -> int:
    if total_budget <= 0:
        return 0
    if total_budget <= 3:
        return total_budget

    planned = int(round(total_budget * settings.seed_query_budget_ratio))
    return max(1, min(total_budget - 1, planned))


def _initial_feedback_size(budget: int, total_tasks: int) -> int:
    if total_tasks <= 0:
        return 0
    target = max(3, min(5, budget // 3))
    return min(total_tasks, target)


def _refinement_targets(budget: int) -> list[int]:
    if budget <= 0:
        return []
    if budget <= 3:
        return list(range(1, budget + 1))

    first = max(1, round(budget * 0.3))
    second = max(first + 1, int(budget * 0.6))
    if second >= budget:
        second = max(first + 1, budget - 1)
    return sorted({first, second, budget})


async def _collect_urls_for_queries(
    lane: LaneSpec,
    queries,
    budget: int,
    seen_urls: set[str],
) -> list[UrlTask]:
    if budget <= 0:
        return []

    candidates: list[CandidateUrl] = []
    candidate_urls: set[str] = set()
    per_query_results = min(
        settings.search_num_results,
        max(settings.search_min_results_per_query, budget * 3),
    )

    for query in queries:
        shaped = shape_query(
            QueryShapeRequest(
                query=query.query,
                suffix=settings.query_shaping_suffix,
                enabled=settings.query_shaping_enabled,
            )
        )
        search_query = shaped.query
        logger.info("%s search (%s): %s", search_provider.provider_name, lane.name, search_query)
        try:
            response = await search_provider.search(
                query=search_query,
                num_results=per_query_results,
            )
        except SearchProviderError:
            continue

        for item in response.results:
            if not item.url:
                continue
            if item.url in seen_urls or item.url in candidate_urls:
                continue
            candidate_urls.add(item.url)
            candidates.append(
                CandidateUrl(
                    url=item.url,
                    title=item.title,
                    source_query=query.query,
                    lane_name=lane.name,
                    score=float(item.score or 0.0),
                    domain=_extract_domain(item.url),
                    title_key=_title_key(item.title),
                    provider_name=search_provider.provider_name,
                    provider_markdown=item.content_markdown,
                    provider_html=item.content_html,
                )
            )

    selected = _rank_candidate_urls(candidates, budget)
    for task in selected:
        seen_urls.add(task.url)
    return selected


def _extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if domain.startswith("www."):
        return domain[4:]
    return domain


def _title_key(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _rank_candidate_urls(candidates: list[CandidateUrl], budget: int) -> list[UrlTask]:
    if budget <= 0 or not candidates:
        return []

    ordered = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    selected: list[CandidateUrl] = []
    backlog: list[CandidateUrl] = []
    selected_urls: set[str] = set()
    selected_domains: set[str] = set()
    selected_titles: set[str] = set()

    for candidate in ordered:
        if candidate.url in selected_urls:
            continue
        if candidate.domain and candidate.domain not in selected_domains:
            selected.append(candidate)
            selected_urls.add(candidate.url)
            selected_domains.add(candidate.domain)
            if candidate.title_key:
                selected_titles.add(candidate.title_key)
            if len(selected) >= budget:
                return [_candidate_to_url_task(item) for item in selected]
            continue
        backlog.append(candidate)

    for candidate in backlog:
        if len(selected) >= budget:
            break
        if candidate.url in selected_urls:
            continue
        if candidate.title_key and candidate.title_key in selected_titles:
            continue
        selected.append(candidate)
        selected_urls.add(candidate.url)
        if candidate.domain:
            selected_domains.add(candidate.domain)
        if candidate.title_key:
            selected_titles.add(candidate.title_key)

    if len(selected) < budget:
        for candidate in backlog:
            if len(selected) >= budget:
                break
            if candidate.url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(candidate.url)

    return [_candidate_to_url_task(item) for item in selected]


def _candidate_to_url_task(candidate: CandidateUrl) -> UrlTask:
    return UrlTask(
        url=candidate.url,
        title=candidate.title,
        source_query=candidate.source_query,
        lane_name=candidate.lane_name,
        provider_name=candidate.provider_name,
        provider_markdown=candidate.provider_markdown,
        provider_html=candidate.provider_html,
    )


async def _store_url_records(run_id: str, url_tasks: list[UrlTask]) -> None:
    records = [
        new_url_record(
            run_id=run_id,
            url=task.url,
            title=task.title,
            source_query=f"{task.lane_name}: {task.source_query}",
        )
        for task in url_tasks
    ]
    await init_db(settings.database_path)
    await _insert_urls(records)


async def _insert_urls(records) -> None:
    from app.services.storage import insert_urls

    await insert_urls(settings.database_path, records)


async def _store_fetched(
    run_id: str,
    url: str,
    html: str,
    run_paths: dict[str, Path],
    source_query: str | None = None,
) -> None:
    html_path = run_paths["html"] / url_to_filename(url, ".html")
    html_path.write_text(html, encoding="utf-8")

    markdown = await html_file_to_markdown(html_path, user_query=source_query)
    markdown_path = run_paths["markdown"] / url_to_filename(url, ".md")
    markdown_path.write_text(markdown, encoding="utf-8")

    await mark_url_fetched(
        settings.database_path,
        run_id=run_id,
        url=url,
        html_path=html_path,
        markdown_path=markdown_path,
    )


async def _store_custom_fetched(
    run_id: str,
    url: str,
    custom_content: CustomContent,
    run_paths: dict[str, Path],
) -> None:
    html_path = run_paths["html"] / url_to_filename(url, ".html")
    html_path.write_text(custom_content.html, encoding="utf-8")

    markdown_path = run_paths["markdown"] / url_to_filename(url, ".md")
    markdown_path.write_text(custom_content.markdown, encoding="utf-8")

    await mark_url_fetched(
        settings.database_path,
        run_id=run_id,
        url=url,
        html_path=html_path,
        markdown_path=markdown_path,
    )


async def _store_provider_fetched(
    run_id: str,
    task: UrlTask,
    run_paths: dict[str, Path],
) -> None:
    markdown = (task.provider_markdown or "").strip()
    if not markdown and not task.provider_html:
        raise MarkdownError(f"{task.provider_name or 'search provider'} returned empty content")

    html_path: Path | None = None
    if task.provider_html:
        html_path = run_paths["html"] / url_to_filename(task.url, ".html")
        html_path.write_text(task.provider_html, encoding="utf-8")

    if not markdown and html_path is not None:
        markdown = await html_file_to_markdown(html_path, user_query=task.source_query)

    markdown_path = run_paths["markdown"] / url_to_filename(task.url, ".md")
    markdown_path.write_text(markdown, encoding="utf-8")

    await mark_url_fetched(
        settings.database_path,
        run_id=run_id,
        url=task.url,
        html_path=html_path,
        markdown_path=markdown_path,
    )


async def _maybe_fetch_custom_content(
    url: str,
    run_paths: dict[str, Path],
) -> CustomContent | None:
    return await asyncio.to_thread(
        fetch_custom_content,
        url,
        run_paths["videos"],
        run_paths["transcripts"],
        run_paths["pdf"],
    )


def build_headful_fallback(playwright, enabled: bool) -> dict:
    """Build headful fallback state."""

    return {
        "enabled": enabled,
        "playwright": playwright,
        "browser": None,
        "context": None,
        "lock": asyncio.Lock(),
    }


async def fetch_with_headful_fallback(
    fallback: dict,
    url: str,
    timeout_ms: int,
) -> str:
    """Fetch HTML using a shared headful browser as a fallback."""

    if not fallback.get("enabled"):
        raise FetchError("Headful fallback disabled")

    async with fallback["lock"]:
        if fallback["browser"] is None:
            fallback["browser"] = await fallback["playwright"].chromium.launch(headless=False)
            fallback["context"] = await fallback["browser"].new_context()
        page = await fallback["context"].new_page()
        try:
            return await capture_html(page, url, timeout_ms=timeout_ms)
        finally:
            await page.close()


async def close_headful_fallback(fallback: dict) -> None:
    """Close any headful fallback browser/context."""

    context = fallback.get("context")
    browser = fallback.get("browser")
    if context is not None:
        await context.close()
    if browser is not None:
        await browser.close()


async def _collect_youtube_transcripts(
    prompt: str,
    videos_dir: Path,
    transcripts_dir: Path,
    max_videos: int,
    model_name: str,
    usage_tracker: UsageTracker | None = None,
) -> list[YouTubeTranscript]:
    """Collect and transcribe YouTube videos in a background thread."""

    if max_videos <= 0:
        return []

    bounded_max = max_videos
    logger.info("Collecting up to %d YouTube videos", bounded_max)

    try:
        transcripts = await asyncio.to_thread(
            transcribe_youtube_videos_with_timeout,
            prompt,
            bounded_max,
            videos_dir,
            transcripts_dir,
            model_name,
            settings.youtube_ingest_timeout_seconds,
        )
        if transcripts and settings.youtube_summarize_transcripts:
            transcripts, summary_usages = await summarize_youtube_transcripts(
                transcripts=transcripts,
                model_name=settings.youtube_summary_model,
                max_chars=settings.youtube_transcript_max_chars,
                concurrency=settings.youtube_summary_concurrency,
            )
            if usage_tracker is not None:
                for usage in summary_usages:
                    await usage_tracker.add(usage, model_name=settings.youtube_summary_model)
        if usage_tracker is not None and transcripts:
            await usage_tracker.add_source("youtube", count=len(transcripts))
        return transcripts
    except TimeoutError:
        logger.warning(
            "YouTube ingestion timed out after %d seconds",
            settings.youtube_ingest_timeout_seconds,
        )
        return []
    except YouTubeError as exc:
        logger.warning("YouTube ingestion failed: %s", exc)
        return []


async def _snapshot_usage(usage_tracker: UsageTracker | None) -> UsageSnapshot | None:
    if usage_tracker is None:
        return None
    return await usage_tracker.snapshot(include_costs=True)


def _log_usage_snapshot(snapshot: UsageSnapshot) -> None:
    logger.info(
        "LLM usage: input=%d output=%d total=%d requests=%d",
        snapshot.input_tokens,
        snapshot.output_tokens,
        snapshot.total_tokens,
        snapshot.requests,
    )
    if snapshot.cost_total is not None:
        logger.info("LLM cost total: $%.6f", snapshot.cost_total)
    if snapshot.cost_unavailable_models:
        logger.info(
            "LLM cost unavailable for models: %s",
            ", ".join(snapshot.cost_unavailable_models),
        )
    if snapshot.sources:
        sources = ", ".join(f"{key}={value}" for key, value in sorted(snapshot.sources.items()))
        logger.info("Source counts: %s", sources)


async def _crawl_tasks(
    run_id: str,
    url_tasks: list[UrlTask],
    context,
    headful_fallback: dict,
    run_paths: dict[str, Path],
    timeout_ms: int,
    usage_tracker: UsageTracker | None,
    reporter: RunReporter | None,
) -> None:
    if not url_tasks:
        return

    concurrency = min(settings.crawl_concurrency_per_lane, len(url_tasks))
    if concurrency <= 1:
        for task in url_tasks:
            await _crawl_single(
                run_id,
                task,
                context,
                headful_fallback,
                run_paths,
                timeout_ms,
                usage_tracker,
                reporter,
            )
        return

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded_crawl(task: UrlTask) -> None:
        async with semaphore:
            await _crawl_single(
                run_id,
                task,
                context,
                headful_fallback,
                run_paths,
                timeout_ms,
                usage_tracker,
                reporter,
            )

    await asyncio.gather(*(_bounded_crawl(task) for task in url_tasks))


async def _crawl_single(
    run_id: str,
    task: UrlTask,
    context,
    headful_fallback: dict,
    run_paths: dict[str, Path],
    timeout_ms: int,
    usage_tracker: UsageTracker | None,
    reporter: RunReporter | None,
) -> None:
    custom_content = await _maybe_fetch_custom_content(task.url, run_paths)
    if custom_content is not None:
        await _store_custom_fetched(run_id, task.url, custom_content, run_paths)
        logger.info("Fetched via custom handler: %s (%s)", task.url, custom_content.source)
        if usage_tracker is not None:
            await usage_tracker.add_source(custom_content.source)
            if custom_content.usage is not None:
                await usage_tracker.add(custom_content.usage, model_name=custom_content.model_name)
        _emit_reporter(reporter, "on_url_done", True)
        return

    if task.provider_markdown or task.provider_html:
        try:
            await _store_provider_fetched(run_id, task, run_paths)
        except MarkdownError as exc:
            logger.warning(
                "Provider content unusable for url: %s (%s)",
                task.url,
                exc,
            )
        else:
            logger.info(
                "Fetched via search provider: %s (%s)",
                task.url,
                task.provider_name or "search",
            )
            if usage_tracker is not None:
                await usage_tracker.add_source(task.provider_name or "search")
            _emit_reporter(reporter, "on_url_done", True)
            return

    page = await context.new_page()
    try:
        logger.debug("Fetching url: %s", task.url)
        html = await capture_html(page, task.url, timeout_ms=timeout_ms)
        await _store_fetched(
            run_id,
            task.url,
            html,
            run_paths,
            source_query=task.source_query,
        )
        if usage_tracker is not None:
            await usage_tracker.add_source("web")
        _emit_reporter(reporter, "on_url_done", True)
        logger.debug("Fetched url: %s", task.url)
    except FetchError as exc:
        if should_retry_headful(exc) and headful_fallback.get("enabled"):
            try:
                html = await fetch_with_headful_fallback(headful_fallback, task.url, timeout_ms)
                await _store_fetched(
                    run_id,
                    task.url,
                    html,
                    run_paths,
                    source_query=task.source_query,
                )
                if usage_tracker is not None:
                    await usage_tracker.add_source("web")
                _emit_reporter(reporter, "on_url_done", True)
                logger.info("Fetched url via headful retry: %s", task.url)
                return
            except FetchError as headful_exc:
                logger.warning(
                    "Headful retry failed for url: %s (%s)",
                    task.url,
                    headful_exc,
                )
                exc = headful_exc
        logger.warning("Failed url: %s (%s)", task.url, exc)
        await mark_url_failed(settings.database_path, run_id=run_id, url=task.url, error=str(exc))
        _emit_reporter(reporter, "on_url_done", False)
    except MarkdownError as exc:
        logger.warning("Failed url: %s (%s)", task.url, exc)
        await mark_url_failed(settings.database_path, run_id=run_id, url=task.url, error=str(exc))
        _emit_reporter(reporter, "on_url_done", False)
    finally:
        await page.close()


def _build_evidence_snippets(url_tasks: list[UrlTask], markdown_dir: Path) -> str:
    snippets: list[str] = []
    for task in url_tasks:
        markdown_path = markdown_dir / url_to_filename(task.url, ".md")
        if not markdown_path.exists():
            continue
        raw = markdown_path.read_text(encoding="utf-8", errors="ignore")
        snippet = raw.strip()
        if not snippet:
            continue
        snippets.append(f"URL: {task.url}\n{snippet}")
    return "\n\n".join(snippets)


def _build_lane_markdown(lane: LaneSpec, url_tasks: list[UrlTask], markdown_dir: Path) -> str:
    parts = [f"# Lane: {lane.name}", lane.goal, ""]
    for task in url_tasks:
        markdown_path = markdown_dir / url_to_filename(task.url, ".md")
        if not markdown_path.exists():
            continue
        raw = markdown_path.read_text(encoding="utf-8", errors="ignore")
        snippet = raw.strip()
        if not snippet:
            continue
        title = task.title or "(untitled)"
        parts.append(f"## {title}\nURL: {task.url}\n\n{snippet}\n")
    return "\n".join(parts)


def _distill_source_text(raw: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    cleaned = raw.strip()
    if not cleaned:
        return ""

    candidates = _extract_signal_segments(cleaned)
    scored: list[tuple[int, str]] = []
    seen_segments: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_segment(candidate)
        if len(normalized) < 20 or normalized in seen_segments:
            continue
        seen_segments.add(normalized)
        score = _score_segment(candidate)
        if score <= 0:
            continue
        scored.append((score, candidate))

    if not scored:
        return " ".join(cleaned.split())[:max_chars].rstrip()

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    highlights: list[str] = []
    quantitative: list[str] = []
    caveats: list[str] = []

    for _, candidate in scored:
        lowered = candidate.lower()
        if any(marker in lowered for marker in CAVEAT_MARKERS):
            caveats.append(candidate)
            continue
        if NUMERIC_SIGNAL_RE.search(candidate):
            quantitative.append(candidate)
            continue
        highlights.append(candidate)

    parts: list[str] = []
    _append_distilled_section(parts, "Highlights", highlights, limit=5)
    _append_distilled_section(parts, "Quantitative Signals", quantitative, limit=4)
    _append_distilled_section(parts, "Caveats", caveats, limit=3)
    distilled = "\n".join(parts).strip()
    if not distilled:
        return " ".join(cleaned.split())[:max_chars].rstrip()
    if len(distilled) <= max_chars:
        return distilled
    return distilled[:max_chars].rstrip()


def _extract_signal_segments(raw: str) -> list[str]:
    lines = [line.strip(" -*\t") for line in raw.splitlines() if line.strip()]
    if len(lines) >= 6:
        return lines
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", raw) if segment.strip()]


def _normalize_segment(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _score_segment(segment: str) -> int:
    lowered = segment.lower()
    if any(marker in lowered for marker in NOISE_MARKERS):
        return -5

    score = 0
    if NUMERIC_SIGNAL_RE.search(segment):
        score += 3
    if any(marker in lowered for marker in CAVEAT_MARKERS):
        score += 2
    if any(marker in lowered for marker in SIGNAL_MARKERS):
        score += 1
    if 40 <= len(segment) <= 280:
        score += 1
    if len(segment) > 450:
        score -= 1
    return score


def _append_distilled_section(
    parts: list[str],
    heading: str,
    values: list[str],
    limit: int,
) -> None:
    if not values:
        return
    parts.append(f"### {heading}")
    for value in values[:limit]:
        parts.append(f"- {value}")
    parts.append("")


def _slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in cleaned.split("-") if part)


def _build_raw_synthesis_context(
    lane_results: list[LaneResult],
    markdown_dir: Path,
    youtube_transcripts: list[YouTubeTranscript],
) -> str:
    parts: list[str] = []
    for lane in lane_results:
        parts.append(f"# Lane: {lane.lane_name}\n{lane.goal}\n")
        for task in lane.url_tasks:
            markdown_path = markdown_dir / url_to_filename(task.url, ".md")
            if not markdown_path.exists():
                continue
            raw = markdown_path.read_text(encoding="utf-8", errors="ignore")
            snippet = raw.strip()
            if not snippet:
                continue
            title = task.title or "(untitled)"
            parts.append(f"## {title}\nURL: {task.url}\n\n{snippet}\n")

    if youtube_transcripts:
        parts.append("# Video Transcripts\n")
        for video in youtube_transcripts:
            title = video.title or "(untitled)"
            snippet = video.transcript.strip()
            if not snippet:
                continue
            parts.append(f"## {title}\nURL: {video.url}\n\n{snippet}\n")
    return "\n".join(parts)


def _build_lane_context_packets(
    prompt: str,
    lane_results: list[LaneResult],
    markdown_dir: Path,
    youtube_transcripts: list[YouTubeTranscript],
) -> list[LaneContextPacket]:
    packets: list[LaneContextPacket] = []
    for lane in lane_results:
        cards = _build_source_cards_for_lane(
            prompt=prompt,
            lane_name=lane.lane_name,
            lane_goal=lane.goal,
            markdown_dir=markdown_dir,
            url_tasks=lane.url_tasks,
        )
        packed_cards = _pack_source_cards(
            cards,
            prompt_builder=lambda source_cards_markdown,
            lane_name=lane.lane_name,
            lane_goal=lane.goal: build_lane_synthesis_prompt(
                prompt=prompt,
                lane_name=lane_name,
                lane_goal=lane_goal,
                source_cards_markdown=source_cards_markdown,
            ),
            max_target_tokens=settings.synthesis_merge_target_tokens,
            max_hard_tokens=settings.synthesis_merge_hard_max_tokens,
            max_sources=settings.synthesis_merge_max_sources,
        )
        if packed_cards:
            packets.append(
                LaneContextPacket(
                    lane_name=lane.lane_name,
                    lane_goal=lane.goal,
                    cards=packed_cards,
                )
            )

    video_cards = _build_youtube_source_cards(prompt, youtube_transcripts)
    packed_video_cards = _pack_source_cards(
        video_cards,
        prompt_builder=lambda source_cards_markdown: build_lane_synthesis_prompt(
            prompt=prompt,
            lane_name="Video Evidence",
            lane_goal="Supplementary evidence from YouTube transcripts",
            source_cards_markdown=source_cards_markdown,
        ),
        max_target_tokens=settings.synthesis_merge_target_tokens,
        max_hard_tokens=settings.synthesis_merge_hard_max_tokens,
        max_sources=settings.synthesis_merge_max_sources,
    )
    if packed_video_cards:
        packets.append(
            LaneContextPacket(
                lane_name="Video Evidence",
                lane_goal="Supplementary evidence from YouTube transcripts",
                cards=packed_video_cards,
            )
        )
    return packets


def _build_followup_memory(
    run_id: str,
    prompt: str,
    synthesis_markdown: str,
    lane_results: list[LaneResult],
    markdown_dir: Path,
    youtube_transcripts: list[YouTubeTranscript],
) -> FollowupMemory:
    source_cards: list[FollowupSourceCard] = []
    for lane in lane_results:
        cards = _build_source_cards_for_lane(
            prompt=prompt,
            lane_name=lane.lane_name,
            lane_goal=lane.goal,
            markdown_dir=markdown_dir,
            url_tasks=lane.url_tasks,
        )
        source_cards.extend(FollowupSourceCard.model_validate(asdict(card)) for card in cards)

    source_cards.extend(
        FollowupSourceCard.model_validate(asdict(card))
        for card in _build_youtube_source_cards(prompt, youtube_transcripts)
    )
    return FollowupMemory(
        run_id=run_id,
        prompt=prompt,
        synthesis_markdown=synthesis_markdown,
        source_cards=source_cards,
    )


def _persist_followup_artifacts(
    run_dir: Path,
    run_id: str,
    prompt: str,
    synthesis_markdown: str,
    lane_results: list[LaneResult],
    markdown_dir: Path,
    youtube_transcripts: list[YouTubeTranscript],
) -> None:
    memory = _build_followup_memory(
        run_id=run_id,
        prompt=prompt,
        synthesis_markdown=synthesis_markdown,
        lane_results=lane_results,
        markdown_dir=markdown_dir,
        youtube_transcripts=youtube_transcripts,
    )
    memory_path = run_dir / FOLLOWUP_MEMORY_FILENAME
    memory_path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")

    transcripts_path = run_dir / YOUTUBE_TRANSCRIPTS_FILENAME
    transcripts_path.write_text(
        json.dumps(
            [item.model_dump() for item in youtube_transcripts], ensure_ascii=True, indent=2
        ),
        encoding="utf-8",
    )


def _build_source_cards_for_lane(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    markdown_dir: Path,
    url_tasks: list[UrlTask],
) -> list[SourceCard]:
    cards: list[SourceCard] = []
    for task in url_tasks:
        markdown_path = markdown_dir / url_to_filename(task.url, ".md")
        if not markdown_path.exists():
            continue
        raw = markdown_path.read_text(encoding="utf-8", errors="ignore")
        card = _build_source_card(
            prompt=prompt,
            lane_name=lane_name,
            lane_goal=lane_goal,
            url=task.url,
            title=task.title,
            raw=raw,
            source_query=task.source_query,
        )
        if card is not None:
            cards.append(card)
    return cards


def _build_youtube_source_cards(
    prompt: str,
    transcripts: list[YouTubeTranscript],
) -> list[SourceCard]:
    cards: list[SourceCard] = []
    for transcript in transcripts:
        card = _build_source_card(
            prompt=prompt,
            lane_name="Video Evidence",
            lane_goal="Supplementary evidence from YouTube transcripts",
            url=transcript.url,
            title=transcript.title,
            raw=transcript.transcript,
            source_query=None,
        )
        if card is not None:
            cards.append(card)
    return cards


def _build_source_card(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    url: str,
    title: str | None,
    raw: str,
    source_query: str | None,
) -> SourceCard | None:
    distilled_text = _distill_source_text(raw, max_chars=settings.evidence_max_chars)
    if not distilled_text:
        return None
    source_kind = _infer_source_kind(url)
    return SourceCard(
        lane_name=lane_name,
        lane_goal=lane_goal,
        url=url,
        title=title,
        source_query=source_query,
        source_kind=source_kind,
        distilled_text=distilled_text,
        relevance_score=_score_source_card(
            prompt=prompt,
            lane_name=lane_name,
            lane_goal=lane_goal,
            url=url,
            title=title,
            source_query=source_query,
            distilled_text=distilled_text,
            source_kind=source_kind,
        ),
    )


def _infer_source_kind(url: str) -> str:
    lowered = url.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    if "reddit.com" in lowered:
        return "reddit"
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        return "pdf"
    return "web"


def _tokenize_score_terms(*values: str | None) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        for token in re.findall(r"[a-z0-9]{3,}", value.lower()):
            tokens.add(token)
    return tokens


def _score_source_card(
    prompt: str,
    lane_name: str,
    lane_goal: str,
    url: str,
    title: str | None,
    source_query: str | None,
    distilled_text: str,
    source_kind: str,
) -> int:
    score = 20
    text = " ".join(part for part in (title, source_query, url, distilled_text) if part).lower()
    prompt_overlap = sum(1 for token in _tokenize_score_terms(prompt) if token in text)
    lane_overlap = sum(1 for token in _tokenize_score_terms(lane_name, lane_goal) if token in text)

    score += min(prompt_overlap * 4, 28)
    score += min(lane_overlap * 3, 18)
    if title:
        score += 4
    if "### Quantitative Signals" in distilled_text:
        score += 6
    if "### Caveats" in distilled_text:
        score += 5
    if "### Highlights" in distilled_text:
        score += 3
    if source_kind in {"reddit", "youtube"}:
        score += 2
    if 200 <= len(distilled_text) <= settings.evidence_max_chars:
        score += 4
    return max(1, min(score, 100))


def _estimate_prompt_tokens(prompt_text: str) -> int:
    return estimate_tokens(prompt_text, model_name=settings.synthesizer_model)


def _ordered_source_cards(cards: list[SourceCard]) -> list[SourceCard]:
    score_sorted = sorted(
        cards,
        key=lambda card: (card.relevance_score, card.value_density, len(card.distilled_text)),
        reverse=True,
    )
    density_sorted = sorted(
        cards,
        key=lambda card: (card.value_density, card.relevance_score, -len(card.distilled_text)),
        reverse=True,
    )

    ordered: list[SourceCard] = []
    seen_urls: set[str] = set()
    for candidate in [*score_sorted[:2], *density_sorted]:
        if candidate.url in seen_urls:
            continue
        seen_urls.add(candidate.url)
        ordered.append(candidate)
    return ordered


def _pack_source_cards(
    cards: list[SourceCard],
    prompt_builder,
    max_target_tokens: int,
    max_hard_tokens: int,
    max_sources: int,
) -> list[SourceCard]:
    if not cards or max_target_tokens <= 0 or max_hard_tokens <= 0 or max_sources <= 0:
        return []

    packed: list[SourceCard] = []
    for card in _ordered_source_cards(cards):
        if len(packed) >= max_sources:
            break

        candidate = [*packed, card]
        candidate_prompt = prompt_builder(_format_source_cards(candidate))
        candidate_tokens = _estimate_prompt_tokens(candidate_prompt)
        if candidate_tokens > max_hard_tokens:
            continue
        if packed and candidate_tokens > max_target_tokens:
            continue
        packed = candidate
    return packed


def _pack_merge_supporting_cards(
    prompt: str,
    node_name: str,
    summary_packets: list[LaneSummaryPacket],
    child_summaries_markdown: str,
) -> list[SourceCard]:
    combined_cards = [card for packet in summary_packets for card in packet.cards]
    return _pack_source_cards(
        combined_cards,
        prompt_builder=lambda evidence_markdown: build_merge_synthesis_prompt(
            prompt=prompt,
            node_name=node_name,
            child_summaries_markdown=child_summaries_markdown,
            supporting_evidence_markdown=evidence_markdown,
        ),
        max_target_tokens=settings.synthesis_merge_target_tokens,
        max_hard_tokens=settings.synthesis_merge_hard_max_tokens,
        max_sources=settings.synthesis_merge_max_sources,
    )


def _format_source_cards(cards: list[SourceCard]) -> str:
    return "\n\n".join(_format_source_card(card) for card in cards)


def _format_source_card(card: SourceCard) -> str:
    title = card.title or "(untitled)"
    lines = [
        "### Source Card",
        f"Lane: {card.lane_name}",
        f"Goal: {card.lane_goal}",
        f"Type: {card.source_kind}",
        f"Relevance Score: {card.relevance_score}/100",
        f"Title: {title}",
        f"URL: {card.url}",
    ]
    if card.source_query:
        lines.append(f"Source Query: {card.source_query}")
    lines.extend(["", card.distilled_text])
    return "\n".join(lines)


def _format_summary_packet(packet: LaneSummaryPacket) -> str:
    parts = [
        f"## {packet.lane_name}",
        f"Goal: {packet.lane_goal}",
        "",
        f"Summary: {packet.synthesis.summary}",
        "",
        "Key Findings:",
        *[f"- {finding}" for finding in packet.synthesis.key_findings],
        "",
        "Top Sources:",
        *[
            f"- {source.title or source.url}: {source.url} ({source.notes})"
            for source in packet.synthesis.sources
        ],
    ]
    if packet.synthesis.gaps:
        parts.extend(["", "Gaps:", *[f"- {gap}" for gap in packet.synthesis.gaps]])
    return "\n".join(parts).strip()


def _format_child_summaries(summary_packets: list[LaneSummaryPacket]) -> str:
    return "\n\n".join(_format_summary_packet(packet) for packet in summary_packets)


def _format_evidence_appendix(cards: list[SourceCard]) -> str:
    if not cards:
        return ""
    return "# Evidence Appendix\n\n" + "\n\n".join(_format_source_card(card) for card in cards)


def _prepare_merge_group(
    prompt: str,
    summary_packets: list[LaneSummaryPacket],
    level: int,
    group_index: int,
) -> tuple[str, str, list[SourceCard], str, int]:
    node_name = f"Merge L{level} G{group_index}"
    child_summaries_markdown = _format_child_summaries(summary_packets)
    supporting_cards = _pack_merge_supporting_cards(
        prompt=prompt,
        node_name=node_name,
        summary_packets=summary_packets,
        child_summaries_markdown=child_summaries_markdown,
    )
    supporting_evidence_markdown = _format_source_cards(supporting_cards)
    prompt_text = build_merge_synthesis_prompt(
        prompt=prompt,
        node_name=node_name,
        child_summaries_markdown=child_summaries_markdown,
        supporting_evidence_markdown=supporting_evidence_markdown,
    )
    return (
        node_name,
        child_summaries_markdown,
        supporting_cards,
        supporting_evidence_markdown,
        _estimate_prompt_tokens(prompt_text),
    )


def _group_summary_packets_for_merge(
    prompt: str,
    summary_packets: list[LaneSummaryPacket],
    level: int,
) -> list[list[LaneSummaryPacket]]:
    groups: list[list[LaneSummaryPacket]] = []
    current: list[LaneSummaryPacket] = []

    for packet in summary_packets:
        candidate = [*current, packet]
        _, _, _, _, estimated_tokens = _prepare_merge_group(
            prompt=prompt,
            summary_packets=candidate,
            level=level,
            group_index=len(groups) + 1,
        )
        if current and (
            len(candidate) > settings.synthesis_merge_max_children
            or estimated_tokens > settings.synthesis_merge_target_tokens
        ):
            groups.append(current)
            current = [packet]
            continue
        current = candidate

    if current:
        groups.append(current)
    return groups


async def _merge_summary_tree(
    prompt: str,
    summary_packets: list[LaneSummaryPacket],
    deps: AgentDeps,
    usage_tracker: UsageTracker | None,
    model_name: str | None,
) -> LaneSummaryPacket:
    current = summary_packets
    level = 1

    while len(current) > 1:
        groups = _group_summary_packets_for_merge(prompt, current, level)
        prepared_groups = [
            _prepare_merge_group(
                prompt=prompt,
                summary_packets=group,
                level=level,
                group_index=index,
            )
            for index, group in enumerate(groups, start=1)
        ]
        for node_name, _, _, _, estimated_tokens in prepared_groups:
            logger.info(
                "Prepared merge node %s with estimated input tokens=%d",
                node_name,
                estimated_tokens,
            )

        merged_syntheses = await asyncio.gather(
            *[
                synthesize_merge_node(
                    prompt=prompt,
                    node_name=node_name,
                    child_summaries_markdown=child_summaries_markdown,
                    supporting_evidence_markdown=supporting_evidence_markdown,
                    deps=deps,
                    usage_tracker=usage_tracker,
                    model_name=model_name,
                )
                for node_name, child_summaries_markdown, _, supporting_evidence_markdown, _ in prepared_groups
            ]
        )
        current = [
            LaneSummaryPacket(
                lane_name=node_name,
                lane_goal=f"Merged summaries at level {level}",
                synthesis=synthesis,
                cards=supporting_cards,
            )
            for (node_name, _, supporting_cards, _, _), synthesis in zip(
                prepared_groups,
                merged_syntheses,
                strict=True,
            )
        ]
        level += 1

    return current[0]


def _build_final_synthesis_input(
    prompt: str,
    summary_packets: list[LaneSummaryPacket],
) -> tuple[str, str, int]:
    merged_summary_markdown = _format_child_summaries(summary_packets)
    supporting_cards = _pack_source_cards(
        [card for packet in summary_packets for card in packet.cards],
        prompt_builder=lambda evidence_markdown: build_final_synthesis_prompt(
            prompt=prompt,
            merged_summary_markdown=merged_summary_markdown,
            evidence_appendix_markdown=evidence_markdown,
        ),
        max_target_tokens=settings.synthesis_final_target_tokens,
        max_hard_tokens=settings.synthesis_final_hard_max_tokens,
        max_sources=settings.synthesis_final_max_sources,
    )
    evidence_appendix_markdown = _format_evidence_appendix(supporting_cards)
    prompt_text = build_final_synthesis_prompt(
        prompt=prompt,
        merged_summary_markdown=merged_summary_markdown,
        evidence_appendix_markdown=evidence_appendix_markdown,
    )
    estimated_tokens = _estimate_prompt_tokens(prompt_text)
    if estimated_tokens > settings.synthesis_final_hard_max_tokens:
        evidence_appendix_markdown = ""
        prompt_text = build_final_synthesis_prompt(
            prompt=prompt,
            merged_summary_markdown=merged_summary_markdown,
            evidence_appendix_markdown=evidence_appendix_markdown,
        )
        estimated_tokens = _estimate_prompt_tokens(prompt_text)
    return merged_summary_markdown, evidence_appendix_markdown, estimated_tokens


async def _synthesize(
    prompt: str,
    markdown_dir: Path,
    lane_results: list[LaneResult],
    youtube_transcripts: list[YouTubeTranscript],
    deps: AgentDeps,
    usage_tracker: UsageTracker | None = None,
    model_name: str | None = None,
) -> tuple[str, UsageSnapshot]:
    lane_packets = _build_lane_context_packets(
        prompt, lane_results, markdown_dir, youtube_transcripts
    )

    if lane_packets:
        leaf_syntheses = await asyncio.gather(
            *[
                synthesize_lane(
                    prompt,
                    packet.lane_name,
                    packet.lane_goal,
                    _format_source_cards(packet.cards),
                    deps,
                    usage_tracker=usage_tracker,
                    model_name=model_name,
                )
                for packet in lane_packets
            ]
        )
        leaf_summary_packets = [
            LaneSummaryPacket(
                lane_name=packet.lane_name,
                lane_goal=packet.lane_goal,
                synthesis=synthesis,
                cards=packet.cards,
            )
            for packet, synthesis in zip(lane_packets, leaf_syntheses, strict=True)
        ]
        merged_summary_markdown, evidence_appendix_markdown, estimated_tokens = (
            _build_final_synthesis_input(prompt, leaf_summary_packets)
        )
        if estimated_tokens > settings.synthesis_final_hard_max_tokens:
            root_packet = await _merge_summary_tree(
                prompt=prompt,
                summary_packets=leaf_summary_packets,
                deps=deps,
                usage_tracker=usage_tracker,
                model_name=model_name,
            )
            merged_summary_markdown, evidence_appendix_markdown, estimated_tokens = (
                _build_final_synthesis_input(prompt, [root_packet])
            )
        logger.info("Final synthesis estimated input tokens=%d", estimated_tokens)
    else:
        raw_context = _build_raw_synthesis_context(lane_results, markdown_dir, youtube_transcripts)
        merged_summary_markdown = raw_context
        evidence_appendix_markdown = ""

    synthesis = await synthesize_review(
        prompt,
        merged_summary_markdown,
        evidence_appendix_markdown,
        deps,
        usage_tracker=usage_tracker,
        model_name=model_name,
    )

    lines = [
        "# ReviewBuddy Synthesis",
        "",
        synthesis.summary,
        "",
        "## Key Findings",
        *[f"- {item}" for item in synthesis.key_findings],
        "",
        "## Recommendation",
        synthesis.recommendation,
        "",
        "## Sources",
        *[f"- {source.title or source.url}: {source.url}" for source in synthesis.sources],
    ]

    if synthesis.gaps:
        lines.extend(["", "## Gaps", *[f"- {gap}" for gap in synthesis.gaps]])

    usage_snapshot = await _snapshot_usage(usage_tracker)
    if usage_snapshot:
        lines.extend(
            [
                "",
                "## Usage",
                f"- Input tokens: {usage_snapshot.input_tokens}",
                f"- Output tokens: {usage_snapshot.output_tokens}",
                f"- Total tokens: {usage_snapshot.total_tokens}",
                f"- Requests: {usage_snapshot.requests}",
            ]
        )
        if usage_snapshot.sources:
            source_counts = ", ".join(
                f"{key}={value}" for key, value in sorted(usage_snapshot.sources.items())
            )
            lines.append(f"- Sources: {source_counts}")
        if usage_snapshot.cost_total is not None:
            lines.append(f"- Total cost: ${usage_snapshot.cost_total:.6f}")
        if usage_snapshot.cost_by_model:
            model_costs = ", ".join(
                f"{model}=${cost.total_cost:.6f}"
                for model, cost in sorted(usage_snapshot.cost_by_model.items())
            )
            lines.append(f"- Cost by model: {model_costs}")
        if usage_snapshot.cost_unavailable_models:
            missing = ", ".join(usage_snapshot.cost_unavailable_models)
            lines.append(f"- Cost unavailable for: {missing}")

    return "\n".join(lines), usage_snapshot or UsageSnapshot(0, 0, 0)
