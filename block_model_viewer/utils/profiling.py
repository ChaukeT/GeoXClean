"""
Lightweight profiling utilities used across the application.

Activate profiling by setting the environment variable
`BLOCK_MODEL_PROFILE=1` or by enabling the ProfilingManager at runtime.
When disabled, the utilities add virtually no overhead.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import threading
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Optional, Dict

import logging

logger = logging.getLogger(__name__)


_PROFILE_ENV_FLAG = "BLOCK_MODEL_PROFILE"
_PROFILE_ENABLED_DEFAULT = os.getenv(_PROFILE_ENV_FLAG, "0") == "1"


class ProfilingManager:
    """Global profiling state controller."""

    _enabled = _PROFILE_ENABLED_DEFAULT
    _lock = threading.Lock()

    @classmethod
    def enable(cls) -> None:
        with cls._lock:
            cls._enabled = True
            logger.info("Profiling enabled")

    @classmethod
    def disable(cls) -> None:
        with cls._lock:
            cls._enabled = False
            logger.info("Profiling disabled")

    @classmethod
    def is_enabled(cls) -> bool:
        with cls._lock:
            return cls._enabled


@dataclass
class _ProfileContext(ContextDecorator):
    name: str
    use_profiler: bool = False
    min_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        self.start_time: Optional[float] = None
        self.profiler: Optional[cProfile.Profile] = None

    def __enter__(self):
        if not ProfilingManager.is_enabled():
            return self
        self.start_time = time.perf_counter()
        if self.use_profiler:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not ProfilingManager.is_enabled():
            return False
        elapsed_ms = 0.0
        if self.profiler is not None:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumtime")
            ps.print_stats(40)
            logger.debug("Profiling stats for %s:\n%s", self.name, s.getvalue())
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if elapsed_ms >= self.min_duration_ms:
            logger.info("Profiled %s: %.2f ms", self.name, elapsed_ms)
        else:
            logger.debug("Profiled %s: %.2f ms (below threshold)", self.name, elapsed_ms)
        return False


def profile_section(name: str, *, use_profiler: bool = False, min_duration_ms: float = 0.0):
    """
    Context manager/decorator that times a code block when profiling is enabled.

    Example:
        with profile_section("renderer.generate_meshes"):
            ...
    """
    return _ProfileContext(name=name, use_profiler=use_profiler, min_duration_ms=min_duration_ms)


# STEP 18: Enhanced profiling with section tracking
_section_stats: Dict[str, Dict[str, float]] = {}
_section_lock = threading.Lock()
_active_sections: Dict[str, float] = {}


def start_section(name: str) -> None:
    """
    Start profiling a named section.
    
    Args:
        name: Section name
    """
    if not ProfilingManager.is_enabled():
        return
    
    with _section_lock:
        _active_sections[name] = time.perf_counter()
        if name not in _section_stats:
            _section_stats[name] = {
                'count': 0,
                'total_ms': 0.0,
                'min_ms': float('inf'),
                'max_ms': 0.0
            }


def end_section(name: str) -> None:
    """
    End profiling a named section.
    
    Args:
        name: Section name
    """
    if not ProfilingManager.is_enabled():
        return
    
    with _section_lock:
        if name not in _active_sections:
            logger.warning(f"end_section called for '{name}' without matching start_section")
            return
        
        start_time = _active_sections.pop(name)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        if name not in _section_stats:
            _section_stats[name] = {
                'count': 0,
                'total_ms': 0.0,
                'min_ms': float('inf'),
                'max_ms': 0.0
            }
        
        stats = _section_stats[name]
        stats['count'] += 1
        stats['total_ms'] += elapsed_ms
        stats['min_ms'] = min(stats['min_ms'], elapsed_ms)
        stats['max_ms'] = max(stats['max_ms'], elapsed_ms)
        
        # Log slow sections
        if elapsed_ms >= 100.0:  # Log sections > 100ms
            logger.info(f"Profiled section '{name}': {elapsed_ms:.2f} ms")


def get_profile_stats() -> Dict[str, Dict[str, float]]:
    """
    Get profiling statistics for all sections.
    
    Returns:
        Dict mapping section names to statistics (count, total_ms, avg_ms, min_ms, max_ms)
    """
    with _section_lock:
        result = {}
        for name, stats in _section_stats.items():
            count = stats['count']
            if count > 0:
                result[name] = {
                    'count': count,
                    'total_ms': stats['total_ms'],
                    'avg_ms': stats['total_ms'] / count,
                    'min_ms': stats['min_ms'] if stats['min_ms'] != float('inf') else 0.0,
                    'max_ms': stats['max_ms']
                }
        return result


def reset_profile_stats() -> None:
    """Reset all profiling statistics."""
    with _section_lock:
        _section_stats.clear()
        _active_sections.clear()
