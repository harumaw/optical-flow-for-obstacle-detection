"""Замер времени по стадиям обработки одного кадра."""
from __future__ import annotations

from time import perf_counter


class StageTimer:
    """Использование: begin(); ...; mark("pre"); ...; mark("flow"); t = end()."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = 0.0
        self._last = 0.0

    def begin(self) -> None:
        self._t0 = perf_counter()
        self._last = self._t0
        self.stages = {}

    def mark(self, name: str) -> None:
        now = perf_counter()
        self.stages[name] = (now - self._last) * 1000.0
        self._last = now

    def end(self) -> dict[str, float]:
        self.stages["total"] = (perf_counter() - self._t0) * 1000.0
        return dict(self.stages)
