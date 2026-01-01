"""Usage tracking for Pydantic AI runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai.usage import RunUsage


@dataclass(frozen=True)
class UsageSnapshot:
    """Immutable snapshot of aggregated usage."""

    input_tokens: int
    output_tokens: int
    requests: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """Aggregate token usage across multiple agent runs."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._requests = 0
        self._lock = asyncio.Lock()

    async def add(self, usage: RunUsage | None) -> None:
        """Add a RunUsage record to the totals."""

        if usage is None:
            return
        async with self._lock:
            self._input_tokens += usage.input_tokens
            self._output_tokens += usage.output_tokens
            self._requests += usage.requests

    async def snapshot(self) -> UsageSnapshot:
        """Return an immutable snapshot of totals."""

        async with self._lock:
            return UsageSnapshot(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                requests=self._requests,
            )
