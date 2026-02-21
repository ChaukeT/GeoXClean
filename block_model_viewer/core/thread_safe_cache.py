"""
Thread-Safe Cache Implementation
================================

Provides thread-safe caching utilities for renderer and visualization data.

Phase 6.2 Implementation: Thread-safe wrapper for caches accessed from
multiple threads (main thread, workers, etc.).

Features:
- RLock-based thread safety
- Get/set/update/clear operations
- Expiration support (optional)
- Statistics tracking

Usage:
    from block_model_viewer.core.thread_safe_cache import ThreadSafeCache
    
    cache = ThreadSafeCache()
    cache.set("key", value)
    value = cache.get("key", default=None)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry(Generic[V]):
    """A cached value with metadata."""
    value: V
    created_at: float  # time.time()
    expires_at: Optional[float] = None  # time.time() or None for no expiry
    access_count: int = 0
    last_accessed: float = 0.0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ThreadSafeCache(Generic[K, V]):
    """
    Thread-safe wrapper for cache data.
    
    Phase 6.2 Implementation: Provides thread-safe access to cached data
    used by the renderer and other components.
    
    Features:
    - All operations are thread-safe via RLock
    - LRU eviction when max size exceeded
    - Optional TTL (time-to-live) for entries
    - Statistics for monitoring
    
    Usage:
        cache = ThreadSafeCache(max_size=100)
        cache.set("drillhole_data", big_array)
        data = cache.get("drillhole_data")
    """
    
    def __init__(
        self,
        max_size: int = 100,
        default_ttl_seconds: Optional[float] = None,
    ):
        """
        Initialize thread-safe cache.
        
        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
            default_ttl_seconds: Default time-to-live for entries (None = no expiry)
        """
        self._data: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: K, default: V = None) -> Optional[V]:
        """
        Get a value from cache.
        
        Thread-safe. Returns default if key not found or expired.
        
        Args:
            key: Cache key
            default: Value to return if key not found
        
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return default
            
            entry = self._data[key]
            
            # Check expiration
            if entry.is_expired:
                del self._data[key]
                self._misses += 1
                return default
            
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end (most recently used)
            self._data.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """
        Set a value in cache.
        
        Thread-safe. Will evict LRU entries if max size exceeded.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live (None uses default)
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._data:
                del self._data[key]
            
            # Calculate expiration
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            
            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                last_accessed=time.time(),
            )
            
            # Add to cache
            self._data[key] = entry
            
            # Evict if necessary
            while len(self._data) > self._max_size:
                oldest_key, _ = self._data.popitem(last=False)
                self._evictions += 1
                logger.debug(f"Cache eviction: {oldest_key}")
    
    def update(self, updates: Dict[K, V], ttl_seconds: Optional[float] = None) -> None:
        """
        Update multiple values at once.
        
        Thread-safe batch update.
        
        Args:
            updates: Dict of key-value pairs to update
            ttl_seconds: Time-to-live for all entries
        """
        with self._lock:
            for key, value in updates.items():
                self.set(key, value, ttl_seconds)
    
    def delete(self, key: K) -> bool:
        """
        Delete a key from cache.
        
        Thread-safe.
        
        Args:
            key: Key to delete
        
        Returns:
            True if key was found and deleted
        """
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Thread-safe.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count
    
    def contains(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._data:
                return False
            if self._data[key].is_expired:
                del self._data[key]
                return False
            return True
    
    def keys(self) -> list:
        """Get list of all keys (thread-safe snapshot)."""
        with self._lock:
            # Remove expired entries first
            self._cleanup_expired()
            return list(self._data.keys())
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries. Must be called with lock held."""
        expired_keys = [k for k, v in self._data.items() if v.is_expired]
        for key in expired_keys:
            del self._data[key]
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._data),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }
    
    def __len__(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._data)
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists."""
        return self.contains(key)


# =============================================================================
# DRILLHOLE CACHE
# =============================================================================

class DrillholeCache(ThreadSafeCache):
    """
    Specialized cache for drillhole rendering data.
    
    Provides typed access methods for common drillhole data.
    """
    
    def __init__(self, max_size: int = 50):
        super().__init__(max_size=max_size)
    
    def set_polylines(self, hole_id: str, polyline: Any) -> None:
        """Cache a drillhole polyline."""
        self.set(f"polyline:{hole_id}", polyline)
    
    def get_polylines(self, hole_id: str) -> Optional[Any]:
        """Get cached polyline for a hole."""
        return self.get(f"polyline:{hole_id}")
    
    def set_segments(self, hole_id: str, segments: Any) -> None:
        """Cache segment data for a hole."""
        self.set(f"segments:{hole_id}", segments)
    
    def get_segments(self, hole_id: str) -> Optional[Any]:
        """Get cached segments for a hole."""
        return self.get(f"segments:{hole_id}")
    
    def clear_hole(self, hole_id: str) -> None:
        """Clear all cache entries for a specific hole."""
        with self._lock:
            keys_to_remove = [
                k for k in self._data.keys()
                if str(k).endswith(f":{hole_id}")
            ]
            for key in keys_to_remove:
                del self._data[key]


# =============================================================================
# GEOMETRY CACHE
# =============================================================================

class GeometryCache(ThreadSafeCache):
    """
    Specialized cache for computed geometry (surfaces, meshes).
    """
    
    def __init__(self, max_size: int = 20):
        super().__init__(max_size=max_size)
    
    def set_surface(self, domain: str, surface: Any) -> None:
        """Cache a domain surface."""
        self.set(f"surface:{domain}", surface)
    
    def get_surface(self, domain: str) -> Optional[Any]:
        """Get cached surface for a domain."""
        return self.get(f"surface:{domain}")
    
    def set_mesh(self, name: str, mesh: Any) -> None:
        """Cache a mesh object."""
        self.set(f"mesh:{name}", mesh)
    
    def get_mesh(self, name: str) -> Optional[Any]:
        """Get cached mesh."""
        return self.get(f"mesh:{name}")


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_drillhole_cache: Optional[DrillholeCache] = None
_geometry_cache: Optional[GeometryCache] = None


def get_drillhole_cache() -> DrillholeCache:
    """Get or create the global drillhole cache."""
    global _drillhole_cache
    if _drillhole_cache is None:
        _drillhole_cache = DrillholeCache()
    return _drillhole_cache


def get_geometry_cache() -> GeometryCache:
    """Get or create the global geometry cache."""
    global _geometry_cache
    if _geometry_cache is None:
        _geometry_cache = GeometryCache()
    return _geometry_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    global _drillhole_cache, _geometry_cache
    if _drillhole_cache is not None:
        _drillhole_cache.clear()
    if _geometry_cache is not None:
        _geometry_cache.clear()

