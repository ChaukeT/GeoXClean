"""
scene_bounds.py — Scene bounds calculation and validation utilities.

All functions here are stateless (pure) or accept state as explicit parameters.
They are called by the orchestrator, never the other way around.

WHY SEPARATE?
Bounds logic affects camera reset, overlay positioning, clipping, and export
framing.  Keeping it in one place prevents feedback loops that caused previous
infinite-render bugs.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BoundsT = Tuple[float, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Pure validators
# ---------------------------------------------------------------------------

def is_valid_bounds(bounds: Optional[Tuple[float, ...]]) -> bool:
    """Return ``True`` if *bounds* is a well-formed (xmin<xmax, ...) 6-tuple."""
    if bounds is None or len(bounds) < 6:
        return False
    x0, x1, y0, y1, z0, z1 = bounds[:6]
    return (x1 > x0) and (y1 > y0) and (z1 > z0)


def sanitize_bounds(
    bounds: Optional[Tuple[float, ...]]
) -> Optional[BoundsT]:
    """Return a clean 6-float tuple or ``None`` if the bounds are invalid."""
    if not is_valid_bounds(bounds):
        return None
    return tuple(float(b) for b in bounds[:6])  # type: ignore[return-value]


def union_bounds(bounds_list: List[Tuple]) -> Optional[BoundsT]:
    """
    Compute the axis-aligned union of a list of 6-element bounds tuples.
    Returns ``None`` if the list is empty.
    """
    if not bounds_list:
        return None
    xmin = min(b[0] for b in bounds_list)
    xmax = max(b[1] for b in bounds_list)
    ymin = min(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)
    zmin = min(b[4] for b in bounds_list)
    zmax = max(b[5] for b in bounds_list)
    return (xmin, xmax, ymin, ymax, zmin, zmax)


# ---------------------------------------------------------------------------
# Scene-level bounds computation
# ---------------------------------------------------------------------------

def compute_scene_bounds(
    fixed_bounds: Optional[BoundsT],
    current_model: Any,
    global_shift: Any,             # np.ndarray or None
    active_layers: Dict[str, Any],
    drillhole_hole_actors: Dict[str, Any],
    fixed_scene_bounds: Optional[BoundsT] = None,
) -> Optional[BoundsT]:
    """
    DETERMINISTIC BOUNDS: compute bounds from real data layers only.

    Ignores HUD/overlay actors to prevent camera-feedback loops.

    Priority:
    1. Locked ``fixed_bounds`` (prevents drift once set)
    2. Union of all data layer actors + drillhole actors (always in local coords)
    3. Block-model bounds (fallback, with double-shift guard)
    4. Legacy ``fixed_scene_bounds`` (backward compat)

    NOTE: Actor bounds (step 2) are preferred over ``current_model.bounds``
    (step 3) because actors are always positioned in local coordinates after
    ``_apply_coordinate_transform_to_meshes()`` runs. ``current_model.bounds``
    may already be in local coordinates (e.g. block models built from SGSIM
    output that ran in local coords), so blindly subtracting ``global_shift``
    would cause a double-shift producing large-magnitude wrong bounds that
    corrupt camera clipping and make the scene invisible.

    Parameters
    ----------
    fixed_bounds
        Previously locked bounds (set after first computation).
    current_model
        Active ``BlockModel`` (may be ``None``).
    global_shift
        The coordinate shift vector applied by ``_to_local_precision``.
    active_layers
        ``Renderer.active_layers`` dict.
    drillhole_hole_actors
        ``Renderer._drillhole_hole_actors`` dict.
    fixed_scene_bounds
        Legacy locked bounds attribute (backward compat alias).

    Returns
    -------
    6-tuple (xmin, xmax, ymin, ymax, zmin, zmax) or ``None``.
    """
    # 1. Return locked bounds if available (prevents drift)
    if fixed_bounds is not None:
        return fixed_bounds

    # 2. Compute from DATA LAYERS ONLY (exclude overlays / previews).
    # Actors are always at local coordinates after coordinate transformation,
    # so this is the most reliable source of scene bounds.
    data_layer_types = (
        'blocks', 'drillhole', 'drillholes',
        'geology_surface', 'mesh', 'classification', 'volume',
    )
    data_bounds: List[Tuple] = []

    for layer_data in active_layers.values():
        layer_type = layer_data.get('layer_type', layer_data.get('type', ''))
        if layer_type not in data_layer_types:
            continue
        actor = layer_data.get('actor')
        if actor is None:
            continue
        try:
            b = actor.GetBounds()
            if b and len(b) >= 6 and not all(v == 0 for v in b):
                data_bounds.append(b)
        except Exception:
            pass

    # Include drillhole actors stored separately
    for actor in drillhole_hole_actors.values():
        if actor is None:
            continue
        try:
            b = actor.GetBounds()
            if b and len(b) >= 6 and not all(v == 0 for v in b):
                data_bounds.append(b)
        except Exception:
            pass

    if data_bounds:
        result = union_bounds(data_bounds)
        logger.debug(f"[scene_bounds] Computed scene bounds from actors: {result}")
        return result

    # 3. Fallback: Use block model bounds.
    # DOUBLE-SHIFT GUARD: current_model.bounds may already be in local coords
    # (e.g. when the block model was built from SGSIM output in local space).
    # Only apply global_shift when the bounds appear to be in world/UTM coords,
    # detected by comparing the bounds center magnitude to the shift magnitude.
    # If the center is already small (<50% of shift magnitude) the data is
    # already local and the shift must NOT be re-applied.
    if current_model is not None and hasattr(current_model, 'bounds') and current_model.bounds is not None:
        try:
            bounds = current_model.bounds
            if global_shift is not None:
                s = global_shift
                bound_center_mag = max(
                    abs(bounds[0] + bounds[1]) / 2,
                    abs(bounds[2] + bounds[3]) / 2,
                    abs(bounds[4] + bounds[5]) / 2,
                )
                shift_mag = max(abs(s[0]), abs(s[1]), abs(s[2]))
                if shift_mag > 0 and bound_center_mag > shift_mag * 0.5:
                    # Bounds are in world/UTM coordinates — transform to local
                    bounds = (
                        bounds[0] - s[0], bounds[1] - s[0],
                        bounds[2] - s[1], bounds[3] - s[1],
                        bounds[4] - s[2], bounds[5] - s[2],
                    )
                # else: bounds are already local, return as-is
            return bounds
        except Exception:
            pass

    # 4. Legacy fallback
    if fixed_scene_bounds is not None:
        return fixed_scene_bounds

    return None


def compute_drillhole_bounds(
    drillhole_hole_actors: Dict[str, Any],
) -> Optional[BoundsT]:
    """
    Return axis-aligned bounds covering all drillhole actors.
    Returns ``None`` if there are no actors or none have valid bounds.
    """
    if not drillhole_hole_actors:
        return None
    bounds_list: List[Tuple] = []
    for actor in drillhole_hole_actors.values():
        try:
            b = actor.GetBounds()
            if is_valid_bounds(b):
                bounds_list.append(b[:6])
        except Exception:
            continue
    return union_bounds(bounds_list)
