import asyncio

from pydantic_ai.usage import RunUsage

from app.services.usage_tracker import UsageTracker


def test_usage_tracker_aggregates() -> None:
    tracker = UsageTracker()

    async def _run() -> None:
        await tracker.add(RunUsage(input_tokens=10, output_tokens=5, requests=1))
        await tracker.add(RunUsage(input_tokens=3, output_tokens=7, requests=2))
        snapshot = await tracker.snapshot()
        assert snapshot.input_tokens == 13
        assert snapshot.output_tokens == 12
        assert snapshot.requests == 3
        assert snapshot.total_tokens == 25

    asyncio.run(_run())
