from app.services.url_handlers import (
    RedditComment,
    RedditPost,
    _extract_subreddit_name,
    format_pdf_markdown,
    format_reddit_markdown,
    format_reddit_subreddit_markdown,
    format_youtube_markdown,
    is_pdf_url,
    is_reddit_url,
)


def test_is_reddit_url() -> None:
    assert is_reddit_url("https://www.reddit.com/r/test/comments/abc123")
    assert is_reddit_url("https://old.reddit.com/r/test/comments/abc123")
    assert is_reddit_url("https://reddit.com/r/test/comments/abc123")
    assert is_reddit_url("https://example.com") is False


def test_format_reddit_markdown() -> None:
    comments = [RedditComment(author="alice", body="Great post", score=10)]
    markdown = format_reddit_markdown(
        title="Test Title",
        url="https://reddit.com/r/test/comments/abc123",
        selftext="Body",
        comments=comments,
        max_comment_chars=200,
    )
    assert "Test Title" in markdown
    assert "URL:" in markdown
    assert "Great post" in markdown


def test_format_youtube_markdown() -> None:
    markdown = format_youtube_markdown(
        title="Video Title",
        url="https://youtu.be/abc123",
        transcript="Transcript text",
        max_chars=100,
    )
    assert "Video Title" in markdown
    assert "https://youtu.be/abc123" in markdown


def test_is_pdf_url() -> None:
    assert is_pdf_url("https://example.com/file.pdf")
    assert is_pdf_url("https://example.com/file.pdf?download=1")
    assert is_pdf_url("https://example.com/file.txt") is False


def test_format_pdf_markdown() -> None:
    markdown = format_pdf_markdown(
        title="PDF Summary",
        url="https://example.com/file.pdf",
        summary="Summary text",
        max_chars=200,
    )
    assert "PDF Summary" in markdown
    assert "https://example.com/file.pdf" in markdown


def test_extract_subreddit_name() -> None:
    assert _extract_subreddit_name("https://www.reddit.com/r/coffeeswap/") == "coffeeswap"
    assert _extract_subreddit_name("https://www.reddit.com/r/test/comments/abc123") == "test"
    assert _extract_subreddit_name("https://example.com") is None


def test_format_reddit_subreddit_markdown() -> None:
    posts = [
        RedditPost(
            title="Post One",
            url="https://www.reddit.com/r/test/comments/abc123/post_one/",
            score=10,
            num_comments=5,
            author="alice",
            selftext="Some details",
        )
    ]
    markdown = format_reddit_subreddit_markdown(
        title="r/test",
        url="https://www.reddit.com/r/test/",
        posts=posts,
        max_post_chars=200,
    )
    assert "r/test" in markdown
    assert "Post One" in markdown
    assert "https://www.reddit.com/r/test/" in markdown
