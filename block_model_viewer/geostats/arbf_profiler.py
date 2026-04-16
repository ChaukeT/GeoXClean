"""
ARBF estimation profiler — timing hooks for performance measurement.

Usage:
    prof = EstimationProfiler()
    with prof.time("search"):
        ...
    with prof.time("factorisation"):
        ...
    summary = prof.summary()
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict


class _TimerContext:
    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler: "EstimationProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_TimerContext":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed = time.perf_counter() - self._t0
        self._profiler._accum[self._name] += elapsed
        self._profiler._counts[self._name] += 1


class EstimationProfiler:
    """Lightweight profiler that accumulates named timing sections."""

    def __init__(self) -> None:
        self._accum: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._extras: Dict[str, Any] = {}
        self._t0 = time.perf_counter()

    def time(self, name: str) -> _TimerContext:
        return _TimerContext(self, name)

    def set(self, key: str, value: Any) -> None:
        self._extras[key] = value

    def summary(self) -> Dict[str, Any]:
        total = time.perf_counter() - self._t0
        result = {"total_seconds": round(total, 4)}
        for name in sorted(self._accum):
            result[f"{name}_seconds"] = round(self._accum[name], 4)
            result[f"{name}_count"] = self._counts[name]
        result.update(self._extras)
        return result

    def report_lines(self) -> str:
        s = self.summary()
        total = s.get("total_seconds", 0.0)
        lines = [f"Total: {total:.3f}s"]
        for k, v in sorted(s.items()):
            if k.endswith("_seconds") and k != "total_seconds":
                name = k.replace("_seconds", "")
                pct = (v / total * 100) if total > 0 else 0
                cnt = s.get(f"{name}_count", "")
                lines.append(f"  {name:20s}: {v:8.3f}s  ({pct:5.1f}%)  [{cnt} calls]")
        for k, v in sorted(s.items()):
            if not k.endswith("_seconds") and not k.endswith("_count") and k != "total_seconds":
                lines.append(f"  {k:20s}: {v}")
        return "\n".join(lines)
