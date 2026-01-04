"""Usage tracking for Pydantic AI runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation

from pydantic_ai.usage import RunUsage


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
        self._model_usage: dict[str, RunUsage] = {}
        self._sources: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def add(self, usage: RunUsage | None, model_name: str | None = None) -> None:
        """Add a RunUsage record to the totals."""

        if usage is None:
            return
        async with self._lock:
            self._input_tokens += usage.input_tokens
            self._output_tokens += usage.output_tokens
            self._requests += usage.requests
            if model_name:
                existing = self._model_usage.get(model_name)
                if existing is None:
                    existing = RunUsage()
                existing.incr(usage)
                self._model_usage[model_name] = existing

    async def add_source(self, source: str, count: int = 1) -> None:
        """Add a source count to the totals."""

        if not source or count <= 0:
            return
        normalized = source.strip().lower()
        async with self._lock:
            self._sources[normalized] = self._sources.get(normalized, 0) + count

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
    model_usage: Mapping[str, RunUsage],
) -> tuple[Decimal | None, dict[str, CostSnapshot], list[str]]:
    try:
        from tokonomics import calculate_pydantic_cost
    except ImportError:
        return None, {}, list(model_usage.keys())

    total_cost = Decimal("0")
    cost_by_model: dict[str, CostSnapshot] = {}
    missing_models: list[str] = []
    for model, usage in model_usage.items():
        if usage.input_tokens + usage.output_tokens <= 0:
            continue
        costs = await calculate_pydantic_cost(model=model, usage=usage)
        if costs is None:
            missing_models.append(model)
            continue
        input_cost = _to_decimal(costs.input_cost)
        output_cost = _to_decimal(costs.output_cost)
        total = _to_decimal(getattr(costs, "total_cost", None), default=input_cost + output_cost)
        cost_by_model[model] = CostSnapshot(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total,
        )
        total_cost += total

    if not cost_by_model:
        return None, {}, missing_models
    return total_cost, cost_by_model, missing_models


def _to_decimal(value: object, default: Decimal | None = None) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return default if default is not None else Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default if default is not None else Decimal("0")
