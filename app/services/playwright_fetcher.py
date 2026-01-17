"""Playwright helpers for HTML capture."""

from contextlib import suppress

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError


class FetchError(RuntimeError):
    """Raised when page fetch fails."""


HEADFUL_RETRY_MARKERS = (
    "access denied",
    "forbidden",
    "captcha",
    "robot",
    "cloudflare",
    "blocked",
    "too many requests",
    "net::err_http2_protocol_error",
    "net::err_access_denied",
    "net::err_blocked_by_client",
    "403",
    "429",
)


def should_retry_headful(error: Exception) -> bool:
    """Return True if the error suggests a headful retry might help."""

    message = str(error).lower()
    return any(marker in message for marker in HEADFUL_RETRY_MARKERS)


def _is_http(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


async def capture_html(page: Page, url: str, timeout_ms: int) -> str:
    """Navigate to a page and return HTML.

    Args:
        page: Playwright page instance.
        url: Target URL.
        timeout_ms: Navigation timeout in milliseconds.

    Returns:
        HTML string.
    """

    if not _is_http(url):
        raise FetchError("Only http(s) URLs are supported")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        with suppress(PlaywrightTimeoutError):
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        return await page.content()
    except PlaywrightTimeoutError as exc:
        raise FetchError(f"Timeout fetching {url}") from exc
    except Exception as exc:  # pragma: no cover - Playwright raises many custom errors
        raise FetchError(f"Failed fetching {url}: {exc}") from exc
