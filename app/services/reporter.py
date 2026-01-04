"""Run progress reporting helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RunReporter:
    """Optional callbacks for reporting run progress."""

    on_lanes_planned: Callable[[int], None] | None = None
    on_urls_discovered: Callable[[int], None] | None = None
    on_url_done: Callable[[bool], None] | None = None
    on_lane_done: Callable[[str], None] | None = None
