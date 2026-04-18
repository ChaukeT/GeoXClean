"""
actor_registry.py — Centralised VTK actor lifecycle management.

Replaces the scattered actor dicts (_drillhole_hole_actors, _overlay_actors, etc.)
that currently live in the Renderer class.  All actor add/remove/clear operations
go through this registry, preventing zombie actors and memory leaks.

MIGRATION NOTE:
During the extraction refactor the existing dict attributes on Renderer remain
as aliases.  Once all sub-renderers are wired to the registry those aliases can
be removed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActorRegistry:
    """
    Central registry for all VTK actors in the scene.

    Categories separate actors by domain so bounds calculation, visibility,
    and cleanup can be scoped to a subsystem without touching others.

    Categories
    ----------
    block_model  — block model mesh actors
    drillhole    — drillhole cylinder / collar / label actors
    surface      — geology surfaces, solids, faults, structural features
    overlay      — axes, scale bar, north arrow, ground grid
    legend       — scalar bar / legend widget actors
    debug        — temporary diagnostic actors (cleared between sessions)
    """

    CATEGORIES = ('block_model', 'drillhole', 'surface', 'overlay', 'legend', 'debug')

    def __init__(self) -> None:
        # actor_id → {'actor': vtk_actor, 'category': str, 'metadata': dict}
        self._actors: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        actor_id: str,
        actor: Any,
        category: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Register an actor under a unique ID within a category."""
        if category not in self.CATEGORIES:
            logger.warning(
                f"[ActorRegistry] Unknown category '{category}' for actor '{actor_id}'"
            )
        self._actors[actor_id] = {
            'actor': actor,
            'category': category,
            'metadata': metadata or {},
        }

    def remove(self, actor_id: str) -> Optional[Any]:
        """
        Unregister and return the actor.
        The caller is responsible for removing it from the plotter.
        Returns ``None`` if the ID is not registered.
        """
        entry = self._actors.pop(actor_id, None)
        return entry['actor'] if entry else None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, actor_id: str) -> Optional[Any]:
        """Return the actor for the given ID, or ``None``."""
        entry = self._actors.get(actor_id)
        return entry['actor'] if entry else None

    def has(self, actor_id: str) -> bool:
        """Return ``True`` if the actor ID is registered."""
        return actor_id in self._actors

    def get_all(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Return ``{actor_id: actor}`` for all actors, or for one category.
        """
        if category is None:
            return {k: v['actor'] for k, v in self._actors.items()}
        return {
            k: v['actor']
            for k, v in self._actors.items()
            if v['category'] == category
        }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear_category(self, category: str) -> List[Any]:
        """
        Unregister all actors in a category.
        Returns the list of actors — the caller removes them from the plotter.
        """
        keys = [k for k, v in self._actors.items() if v['category'] == category]
        actors = []
        for key in keys:
            actors.append(self._actors.pop(key)['actor'])
        return actors

    def clear_all(self) -> List[Any]:
        """Unregister every actor and return the full list."""
        actors = [v['actor'] for v in self._actors.values()]
        self._actors.clear()
        return actors

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def get_bounds(self, category: Optional[str] = None) -> Optional[Tuple[float, ...]]:
        """
        Compute combined axis-aligned bounds for all registered actors
        (or for one category).

        Returns ``(xmin, xmax, ymin, ymax, zmin, zmax)`` or ``None`` if no
        actors have valid geometry.  HUD actors at implausible coordinates
        (> 1e15) are excluded.
        """
        actors = self.get_all(category)
        if not actors:
            return None

        xmin = ymin = zmin = float('inf')
        xmax = ymax = zmax = float('-inf')
        found = False

        for actor in actors.values():
            try:
                b = actor.GetBounds()
                if b is None or len(b) < 6:
                    continue
                if any(abs(v) > 1e15 for v in b):
                    continue  # HUD / overlay actor — skip
                xmin = min(xmin, b[0]); xmax = max(xmax, b[1])
                ymin = min(ymin, b[2]); ymax = max(ymax, b[3])
                zmin = min(zmin, b[4]); zmax = max(zmax, b[5])
                found = True
            except Exception:
                pass

        return (xmin, xmax, ymin, ymax, zmin, zmax) if found else None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._actors)

    def __repr__(self) -> str:
        by_cat: Dict[str, int] = {}
        for v in self._actors.values():
            by_cat[v['category']] = by_cat.get(v['category'], 0) + 1
        return f"ActorRegistry({by_cat})"
