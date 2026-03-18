"""Usage tracking for Codex-backed runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal

from app.services.codex_exec import CodexUsage


@dataclass(frozen=True)
class ModelUsageSnapshot:
    """Usage snapshot for a single model."""

    model: str
    input_tokens: int
    output_tokens: int
    requests: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CostSnapshot:
    """Cost snapshot for a single model."""

    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


@dataclass(frozen=True)
class UsageSnapshot:
    """Immutable snapshot of aggregated usage."""

    input_tokens: int
    output_tokens: int
    requests: int
    sources: dict[str, int] = field(default_factory=dict)
    per_model: dict[str, ModelUsageSnapshot] = field(default_factory=dict)
    cost_total: Decimal | None = None
    cost_by_model: dict[str, CostSnapshot] = field(default_factory=dict)
    cost_unavailable_models: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageTracker:
    """Aggregate token usage across multiple agent runs."""

    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._requests = 0
        self._model_usage: dict[str, CodexUsage] = {}
        self._sources: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def add(self, usage: CodexUsage | None, model_name: str | None = None) -> None:
        """Add a usage record to the totals."""

        if usage is None:
            return
        async with self._lock:
            self._merge_usage(usage, model_name)

    def add_nowait(self, usage: CodexUsage | None, model_name: str | None = None) -> None:
        """Add a usage record synchronously for same-thread call sites."""

        if usage is None:
            return
        self._merge_usage(usage, model_name)

    async def add_source(self, source: str, count: int = 1) -> None:
        """Add a source count to the totals."""

        if not source or count <= 0:
            return
        normalized = source.strip().lower()
        async with self._lock:
            self._sources[normalized] = self._sources.get(normalized, 0) + count

    def _merge_usage(self, usage: CodexUsage, model_name: str | None) -> None:
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._requests += usage.requests
        if not model_name:
            return
        existing = self._model_usage.get(model_name)
        if existing is None:
            existing = CodexUsage(requests=0)
        self._model_usage[model_name] = CodexUsage(
            input_tokens=existing.input_tokens + usage.input_tokens,
            output_tokens=existing.output_tokens + usage.output_tokens,
            cached_input_tokens=existing.cached_input_tokens + usage.cached_input_tokens,
            requests=existing.requests + usage.requests,
        )

    async def snapshot(self, include_costs: bool = False) -> UsageSnapshot:
        """Return an immutable snapshot of totals."""

        async with self._lock:
            sources = dict(self._sources)
            model_usage = dict(self._model_usage)
            snapshot = UsageSnapshot(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                requests=self._requests,
                sources=sources,
                per_model={
                    model: ModelUsageSnapshot(
                        model=model,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        requests=usage.requests,
                    )
                    for model, usage in model_usage.items()
                },
            )

        if not include_costs or not model_usage:
            return snapshot

        cost_total, cost_by_model, missing_models = await _calculate_costs(model_usage)
        return replace(
            snapshot,
            cost_total=cost_total,
            cost_by_model=cost_by_model,
            cost_unavailable_models=tuple(missing_models),
        )

    async def snapshot_basic(self) -> UsageSnapshot:
        """Return an immutable snapshot without cost calculation."""

        return await self.snapshot(include_costs=False)


async def _calculate_costs(
    model_usage: Mapping[str, CodexUsage],
) -> tuple[Decimal | None, dict[str, CostSnapshot], list[str]]:
    unavailable = list(model_usage.keys())
    return None, {}, unavailable
