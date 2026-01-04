"""Custom URL handlers (e.g., Reddit, YouTube)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from praw.exceptions import InvalidURL
from pydantic_ai.usage import RunUsage

from app.core.settings import get_settings
from app.services.storage import url_to_filename
from app.services.youtube_transcriber import YouTubeError, is_youtube_url

logger = logging.getLogger(__name__)
settings = get_settings()

REDDIT_DOMAINS = {"reddit.com", "www.reddit.com", "old.reddit.com"}
PDF_MEDIA_TYPE = "application/pdf"


@dataclass(frozen=True)
class CustomContent:
    """Custom handler output."""

    html: str
    markdown: str
    source: str
    usage: RunUsage | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class RedditComment:
    author: str | None
    body: str
    score: int | None


@dataclass(frozen=True)
class RedditPost:
    title: str
    url: str
    score: int | None
    num_comments: int | None
    author: str | None
    selftext: str | None


def is_reddit_url(url: str) -> bool:
    """Return True if the URL is a Reddit URL."""

    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() in REDDIT_DOMAINS
    except Exception:
        return False


def is_pdf_url(url: str) -> bool:
    """Return True if the URL looks like a PDF."""

    try:
        return urlparse(url).path.lower().endswith(".pdf")
    except Exception:
        return False


def format_reddit_markdown(
    title: str,
    url: str,
    selftext: str | None,
    comments: Iterable[RedditComment],
    max_comment_chars: int,
) -> str:
    """Render Reddit content to markdown."""

    lines = [f"# {title}", f"URL: {url}", ""]
    if selftext:
        lines.append(selftext.strip())
        lines.append("")

    lines.append("## Top Comments")
    for comment in comments:
        body = comment.body.strip()
        if max_comment_chars > 0:
            body = body[:max_comment_chars]
        author = comment.author or "unknown"
        score = comment.score if comment.score is not None else "n/a"
        lines.append(f"- **{author}** (score: {score}): {body}")

    return "\n".join(lines).strip()


def format_youtube_markdown(
    title: str | None,
    url: str,
    transcript: str,
    max_chars: int,
) -> str:
    """Render YouTube transcript to markdown."""

    header = title or "YouTube Transcript"
    snippet = transcript.strip()
    if max_chars > 0:
        snippet = snippet[:max_chars]
    return f"# {header}\nURL: {url}\n\n{snippet}".strip()


def format_pdf_markdown(
    title: str,
    url: str,
    summary: str,
    max_chars: int,
) -> str:
    """Render PDF summary to markdown."""

    snippet = summary.strip()
    if max_chars > 0:
        snippet = snippet[:max_chars]
    return f"# {title}\nURL: {url}\n\n{snippet}".strip()


def format_reddit_subreddit_markdown(
    title: str,
    url: str,
    posts: Iterable[RedditPost],
    max_post_chars: int,
) -> str:
    """Render subreddit listing to markdown."""

    lines = [f"# {title}", f"URL: {url}", "", "## Top Posts"]
    for post in posts:
        score = post.score if post.score is not None else "n/a"
        comments = post.num_comments if post.num_comments is not None else "n/a"
        lines.append(
            f"- **{post.title}** (score: {score}, comments: {comments}) {post.url}"
        )
        if post.selftext:
            snippet = post.selftext.strip()
            if max_post_chars > 0:
                snippet = snippet[:max_post_chars]
            if snippet:
                lines.append(f"  {snippet}")

    return "\n".join(lines).strip()


def _get_reddit_client() -> Any | None:
    if not settings.reddit_client_id or not settings.reddit_client_secret:
        logger.warning("Reddit credentials not configured; skipping Reddit handler")
        return None

    try:
        import praw
    except ImportError as exc:
        logger.warning("praw not installed; skipping Reddit handler (%s)", exc)
        return None

    reddit_kwargs: dict[str, Any] = {
        "client_id": settings.reddit_client_id,
        "client_secret": settings.reddit_client_secret,
        "user_agent": settings.reddit_user_agent or "reviewbuddy/1.0",
        "check_for_updates": False,
        "timeout": 30,
    }

    if not settings.reddit_read_only:
        if settings.reddit_username and settings.reddit_password:
            reddit_kwargs.update(
                username=settings.reddit_username,
                password=settings.reddit_password,
            )
        else:
            logger.warning("REDDIT_READ_ONLY=false but credentials missing; using read-only")

    reddit = praw.Reddit(**reddit_kwargs)
    reddit.read_only = settings.reddit_read_only or not (
        settings.reddit_username and settings.reddit_password
    )
    return reddit


def _extract_subreddit_name(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "r":
        return parts[1]
    return None


def _fetch_reddit_subreddit_content(client: Any, url: str) -> CustomContent | None:
    subreddit_name = _extract_subreddit_name(url)
    if not subreddit_name:
        return None

    subreddit = client.subreddit(subreddit_name)
    posts: list[RedditPost] = []
    for submission in subreddit.hot(limit=settings.reddit_post_limit):
        posts.append(
            RedditPost(
                title=submission.title or "Untitled",
                url=f"https://www.reddit.com{submission.permalink}",
                score=getattr(submission, "score", None),
                num_comments=getattr(submission, "num_comments", None),
                author=getattr(getattr(submission, "author", None), "name", None),
                selftext=getattr(submission, "selftext", None),
            )
        )
        if len(posts) >= settings.reddit_post_limit:
            break

    if not posts:
        return None

    markdown = format_reddit_subreddit_markdown(
        title=f"r/{subreddit_name}",
        url=url,
        posts=posts,
        max_post_chars=settings.reddit_comment_max_chars,
    )
    html = f"<pre>{markdown}</pre>"
    return CustomContent(html=html, markdown=markdown, source="reddit")


def fetch_reddit_content(url: str) -> CustomContent | None:
    """Fetch Reddit thread via API and render markdown."""

    client = _get_reddit_client()
    if client is None:
        return None

    try:
        submission = client.submission(url=url)
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)
    except InvalidURL:
        return _fetch_reddit_subreddit_content(client, url)

    comments: list[RedditComment] = []
    for comment in submission.comments.list():
        body = getattr(comment, "body", "")
        if not body:
            continue
        comments.append(
            RedditComment(
                author=getattr(getattr(comment, "author", None), "name", None),
                body=body,
                score=getattr(comment, "score", None),
            )
        )
        if len(comments) >= settings.reddit_comment_limit:
            break

    markdown = format_reddit_markdown(
        title=submission.title or "Reddit Thread",
        url=url,
        selftext=getattr(submission, "selftext", None),
        comments=comments,
        max_comment_chars=settings.reddit_comment_max_chars,
    )
    html = f"<pre>{markdown}</pre>"
    return CustomContent(html=html, markdown=markdown, source="reddit")


def fetch_youtube_content(
    url: str,
    videos_dir: Path,
    transcripts_dir: Path,
) -> CustomContent | None:
    """Fetch YouTube transcript via local Whisper."""

    if not is_youtube_url(url):
        return None

    from app.services.youtube_transcriber import download_audio, transcribe_audio

    try:
        audio_path, title = download_audio(url, videos_dir)
        transcript = transcribe_audio(audio_path, settings.whisper_model)
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcripts_dir / f"{audio_path.stem}.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
    except YouTubeError as exc:
        logger.warning("YouTube handler failed: %s", exc)
        return None

    markdown = format_youtube_markdown(
        title=title,
        url=url,
        transcript=transcript,
        max_chars=settings.youtube_transcript_max_chars,
    )
    html = f"<pre>{markdown}</pre>"
    return CustomContent(html=html, markdown=markdown, source="youtube")


def _summarize_pdf_with_gemini(pdf_path: Path, source_url: str) -> tuple[str, RunUsage | None]:
    from pydantic_ai import Agent, DocumentUrl
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    provider = GoogleProvider(api_key=settings.google_api_key)
    model = GoogleModel(settings.pdf_model_name, provider=provider)
    agent = Agent(
        model,
        system_prompt=(
            "You are a research assistant summarizing PDFs for product reviews. "
            "Extract key findings, caveats, and any quantitative specs."
        ),
    )

    file_ref = provider.client.files.upload(file=str(pdf_path))
    document = DocumentUrl(url=file_ref.uri, media_type=file_ref.mime_type or PDF_MEDIA_TYPE)
    prompt = (
        "Summarize this PDF for a product research dossier. Provide key points, "
        "relevant specs, and any cautions. Source URL: "
        f"{source_url}"
    )
    result = agent.run_sync([prompt, document])
    return result.output, result.usage()


def fetch_pdf_content(url: str, pdf_dir: Path) -> CustomContent | None:
    """Download and summarize a PDF via Gemini."""

    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.warning("PDF download failed: %s", exc)
        return None

    content_type = response.headers.get("content-type", "").lower()
    if PDF_MEDIA_TYPE not in content_type and not is_pdf_url(url):
        return None

    content = response.content
    if len(content) > settings.pdf_max_bytes:
        logger.warning("PDF too large (%d bytes): %s", len(content), url)
        return None

    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / url_to_filename(url, ".pdf")
    pdf_path.write_bytes(content)

    summary, usage = _summarize_pdf_with_gemini(pdf_path, url)
    markdown = format_pdf_markdown(
        title="PDF Summary",
        url=url,
        summary=summary,
        max_chars=settings.pdf_summary_max_chars,
    )
    html = f"<pre>{markdown}</pre>"
    return CustomContent(
        html=html,
        markdown=markdown,
        source="pdf",
        usage=usage,
        model_name=settings.pdf_model_name,
    )


def fetch_custom_content(
    url: str,
    videos_dir: Path,
    transcripts_dir: Path,
    pdf_dir: Path,
) -> CustomContent | None:
    """Route URL to a custom handler if supported."""

    if is_reddit_url(url):
        return fetch_reddit_content(url)
    if is_youtube_url(url):
        return fetch_youtube_content(url, videos_dir, transcripts_dir)
    if is_pdf_url(url):
        return fetch_pdf_content(url, pdf_dir)
    return None
