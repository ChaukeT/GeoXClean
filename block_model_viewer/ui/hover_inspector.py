"""
Click Inspect — data-tip overlay for the 3D viewport.

Provides:
- TooltipOverlay: styled QFrame floating near the cursor
- ClickInspectorController: click-to-inspect with tooltip + highlight support
- Pick adapters for block models, drillholes, and surfaces

All heavy picking is delegated to PickingController.  For user-initiated
clicks, the LOD cell-level gate is bypassed (vtkCellPicker on 500k cells
takes ~50-150ms which is acceptable for click latency).
Data extraction from adapters is O(1) via precomputed numpy arrays.

Drillhole picking uses world-position matching to correctly identify
the clicked segment, since drillholes are rendered as tube meshes
whose cell indices don't match original line segment indices.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..visualization.picking_controller import get_picking_controller

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PickResult:
    """Immutable result from a pick operation."""

    hit: bool
    layer_name: str = ""
    layer_type: str = ""  # block_model | drillhole | surface | generic
    identifier: str = ""  # block index, hole ID, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def miss() -> "PickResult":
        return PickResult(hit=False)


# ---------------------------------------------------------------------------
# Pick adapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PickAdapter(Protocol):
    def can_handle(self, actor: Any) -> bool: ...
    def extract_hover_data(self, actor: Any) -> PickResult: ...
    def extract_cell_data(self, actor: Any, cell_id: int,
                          pick_pos: Optional[Tuple[float, float, float]] = None) -> PickResult: ...
    def invalidate_cache(self) -> None: ...


# ---------------------------------------------------------------------------
# Block model adapter
# ---------------------------------------------------------------------------

class BlockModelPickAdapter:
    """O(1) block model pick data extraction via precomputed numpy arrays.

    FIX (Issue P2): Tracks the actor identity so the cache auto-
    invalidates when the mesh_actor is replaced (e.g., after
    set_property_coloring fallback rebuilds the actor).  Without
    this, a stale cache from the old actor would return wrong
    property values for the new mesh.
    """

    # I1+I2 FIX: Internal / VTK-generated arrays that must NEVER appear in
    # the tooltip.  "domain_mask_colors" is a 4-component RGBA uint8 array
    # added by apply_domain_mask_transparency() — reading it with .item()
    # crashes (ValueError: can only convert array of size 1).  "domain_mask"
    # is a 0/1 visibility flag, not a geological property.
    _INTERNAL_ARRAY_NAMES: frozenset = frozenset({
        "Original_ID",
        "domain_mask",
        "domain_mask_colors",
    })

    @staticmethod
    def _is_internal_array(name: str) -> bool:
        """Check if an array name is internal / VTK-generated."""
        if name in BlockModelPickAdapter._INTERNAL_ARRAY_NAMES:
            return True
        # VTK convention: internal arrays start with "vtk"
        if name.startswith("vtk"):
            return True
        return False

    def __init__(self, renderer: Any):
        self._renderer = renderer
        self._original_ids: Optional[np.ndarray] = None
        self._prop_arrays: Dict[str, np.ndarray] = {}
        self._prop_names: List[str] = []
        self._cached = False
        self._cached_actor_id: Optional[int] = None  # id() of the actor we cached from

    def can_handle(self, actor: Any) -> bool:
        return (
            actor is not None
            and self._renderer.mesh_actor is not None
            and actor is self._renderer.mesh_actor
        )

    def extract_hover_data(self, actor: Any) -> PickResult:
        return PickResult(
            hit=True,
            layer_name="Block Model",
            layer_type="block_model",
            identifier="",
        )

    def extract_cell_data(self, actor: Any, cell_id: int,
                          pick_pos: Optional[Tuple[float, float, float]] = None) -> PickResult:
        if not self._ensure_cache():
            return PickResult.miss()

        if self._original_ids is None or cell_id < 0 or cell_id >= len(self._original_ids):
            return PickResult.miss()

        block_id = int(self._original_ids[cell_id])
        # FIX (Issue #5): Check for dtype.min sentinel (not just < 0),
        # since legitimate domain codes may use negative values.
        # The sentinel for int64 is -9223372036854775808.
        if block_id < -2_000_000_000:  # Well below any realistic block index
            return PickResult.miss()

        props: Dict[str, Any] = {}
        for name in self._prop_names:
            arr = self._prop_arrays.get(name)
            if arr is not None and cell_id < len(arr):
                val = arr[cell_id]
                # I3 FIX: Guard against multi-component arrays (e.g.,
                # domain_mask_colors is (N, 4) RGBA uint8).  Indexing a
                # multi-component VTK array returns a numpy array of shape
                # (n_components,), and calling .item() on it raises
                # ValueError.  Skip any value that isn't a scalar.
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    continue
                if isinstance(val, (np.floating, float)) and np.isnan(val):
                    continue
                # FIX (Issue #21): Skip integer sentinel values (dtype.min)
                # from ImageData empty cells. Without this, hovering over
                # empty cells shows meaningless huge negative numbers.
                # R9-01 FIX: dtype-aware sentinel check. After int8/int16
                # downcasting, sentinels are -128 or -32768, not just -2B.
                if isinstance(val, (np.integer, int)):
                    int_val = int(val)
                    if hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.integer):
                        _sent = int(np.iinfo(val.dtype).min)
                    else:
                        _sent = -2_000_000_000
                    # FP-11 FIX: Use strict equality (==) instead of <=
                    # to avoid rejecting legitimate values that happen to
                    # equal dtype.min (e.g., -128 for int8 domain codes).
                    if int_val == _sent:
                        continue
                props[name] = val.item() if hasattr(val, "item") else val

        return PickResult(
            hit=True,
            layer_name="Block Model",
            layer_type="block_model",
            identifier=str(block_id),
            properties=props,
            world_pos=pick_pos or (0.0, 0.0, 0.0),
        )

    def invalidate_cache(self) -> None:
        self._original_ids = None
        self._prop_arrays.clear()
        self._prop_names.clear()
        self._cached = False
        self._cached_actor_id = None

    # -- internal --

    def _ensure_cache(self) -> bool:
        actor = self._renderer.mesh_actor
        if actor is None:
            self._cached = False
            return False

        # FIX (Issue P2): Auto-invalidate if the actor changed since
        # last cache build (e.g., after set_property_coloring fallback
        # rebuilds the mesh actor).
        current_actor_id = id(actor)
        # FP-13 FIX: Also check mesh cell count to detect in-place mutations
        # (e.g., extract_cells modifying the mapper input without replacing
        # the actor object).  id() alone misses this case.
        _n_cells = 0
        try:
            _mapper = actor.GetMapper()
            if _mapper and _mapper.GetInput():
                _n_cells = _mapper.GetInput().GetNumberOfCells()
        except Exception:
            pass
        _cached_n = getattr(self, '_cached_n_cells', -1)
        if self._cached and self._cached_actor_id == current_actor_id and _n_cells == _cached_n:
            return True
        # Actor changed or first build — rebuild cache
        if self._cached and self._cached_actor_id != current_actor_id:
            logger.debug("BlockModelPickAdapter: actor changed, rebuilding cache")
            self._cached = False

        mapper = actor.GetMapper()
        if mapper is None:
            return False
        vtk_data = mapper.GetInput()
        if vtk_data is None:
            return False

        try:
            import pyvista as pv

            mesh = pv.wrap(vtk_data)
        except Exception:
            return False

        # E1 FIX: Guard cell_data access against VTK data becoming
        # unavailable between pv.wrap() and dictionary reads (e.g., if
        # the mapper input is swapped on another thread during a scene
        # update).  Without this, a stale or corrupt VTK dataset could
        # crash the tooltip instead of returning a safe miss.
        try:
            if "Original_ID" in mesh.cell_data:
                self._original_ids = mesh.cell_data["Original_ID"]
            else:
                self._original_ids = np.arange(mesh.n_cells, dtype=np.int64)

            # I1+I2 FIX: Exclude internal / VTK-generated arrays from tooltip
            # properties.  "domain_mask_colors" is a 4-component RGBA array
            # that crashes .item(); "domain_mask" is a 0/1 flag, not a
            # geological property; "vtk*" arrays are VTK internals.
            self._prop_names = [
                k for k in mesh.cell_data.keys()
                if not self._is_internal_array(k)
            ]
            self._prop_arrays = {k: mesh.cell_data[k] for k in self._prop_names}
        except Exception as exc:
            logger.debug("BlockModelPickAdapter: cell_data access failed: %s", exc)
            return False
        self._cached = True
        self._cached_actor_id = current_actor_id
        self._cached_n_cells = _n_cells  # FP-13: track cell count for mutation detection
        return True


# ---------------------------------------------------------------------------
# Drillhole adapter
# ---------------------------------------------------------------------------

class DrillholePickAdapter:
    """Pick adapter for drillholes.

    Drillholes are rendered as tube meshes (merged into a single actor).
    The tube mesh cell indices do NOT correspond to original line segment
    indices, so we use the pick world position to find the nearest segment.
    """

    def __init__(self, renderer: Any):
        self._renderer = renderer
        self._actor_id_to_hole: Dict[int, str] = {}
        self._is_merged: bool = False
        self._merged_actor_id: Optional[int] = None

    def can_handle(self, actor: Any) -> bool:
        self._ensure_cache()
        aid = id(actor)
        if aid in self._actor_id_to_hole:
            return True
        if self._is_merged and aid == self._merged_actor_id:
            return True
        return False

    def extract_hover_data(self, actor: Any) -> PickResult:
        """Actor-level data — includes collar coords when available."""
        aid = id(actor)
        hole_id = self._actor_id_to_hole.get(aid, "")

        if hole_id:
            props = self._get_hole_basic_props(hole_id)
            return PickResult(
                hit=True,
                layer_name=f"Drillhole: {hole_id}",
                layer_type="drillhole",
                identifier=hole_id,
                properties=props,
            )

        if self._is_merged and aid == self._merged_actor_id:
            return PickResult(
                hit=True,
                layer_name="Drillholes",
                layer_type="drillhole",
                identifier="",
            )
        return PickResult.miss()

    def extract_cell_data(self, actor: Any, cell_id: int,
                          pick_pos: Optional[Tuple[float, float, float]] = None) -> PickResult:
        """Segment-level data using cell_data metadata or world position fallback.

        First tries cell_data["_pick_hole_hash"] on the mesh (reliable, survives
        merge).  Falls back to world-position nearest-segment search.
        """
        layer_data = self._get_drillhole_layer_data()

        # --- PRIMARY: cell_data metadata lookup (reliable) ---
        if cell_id >= 0 and actor is not None:
            try:
                import pyvista as _pv
                mesh = None
                # Get mesh from actor's mapper
                mapper = actor.GetMapper()
                if mapper is not None:
                    ds = mapper.GetInput()
                    if ds is not None:
                        mesh = _pv.wrap(ds)
                if mesh is not None and "_pick_hole_hash" in mesh.cell_data:
                    hole_hash = int(mesh.cell_data["_pick_hole_hash"][cell_id])
                    hash_map = getattr(self._renderer, "_pick_hash_to_hole", {})
                    hole_id = hash_map.get(hole_hash, "")
                    if hole_id and layer_data is not None:
                        # Found hole via cell metadata — now find nearest segment
                        # using pick_pos but ONLY within this hole
                        props: Dict[str, Any] = {"hole_id": hole_id}
                        seg_idx = -1
                        if pick_pos is not None:
                            hole_polys = layer_data.get("hole_polys", {})
                            poly = hole_polys.get(hole_id)
                            if poly is not None and hasattr(poly, "points") and poly.n_points >= 2:
                                pts = np.asarray(poly.points)
                                dists = self._point_to_segments_dist(
                                    np.array(pick_pos, dtype=float),
                                    pts[:-1], pts[1:],
                                )
                                seg_idx = int(np.argmin(dists))

                        # Populate interval data
                        seg_lith = layer_data.get("hole_segment_lith", {}).get(hole_id, [])
                        seg_assay = layer_data.get("hole_segment_assay", {}).get(hole_id, [])
                        seg_from = layer_data.get("hole_segment_from_depth", {}).get(hole_id, [])
                        seg_to = layer_data.get("hole_segment_to_depth", {}).get(hole_id, [])

                        if 0 <= seg_idx < len(seg_lith) and seg_lith[seg_idx]:
                            props["lithology"] = seg_lith[seg_idx]
                        if 0 <= seg_idx < len(seg_assay):
                            val = seg_assay[seg_idx]
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                assay_field = layer_data.get("assay_field", "grade")
                                props[assay_field] = round(float(val), 4)
                        if 0 <= seg_idx < len(seg_from):
                            props["from"] = round(float(seg_from[seg_idx]), 2)
                        if 0 <= seg_idx < len(seg_to):
                            props["to"] = round(float(seg_to[seg_idx]), 2)

                        collar = layer_data.get("collar_coords", {}).get(hole_id)
                        if collar:
                            props["collar"] = f"({collar[0]:.1f}, {collar[1]:.1f}, {collar[2]:.1f})"
                        props["_segment_idx"] = seg_idx

                        logger.debug("DrillholePick (cell_data): %s[%d]", hole_id, seg_idx)
                        return PickResult(
                            hit=True,
                            layer_name=f"Drillhole: {hole_id}",
                            layer_type="drillhole",
                            identifier=hole_id,
                            properties=props,
                            world_pos=pick_pos,
                        )
            except Exception as exc:
                logger.debug("Cell_data pick failed, falling back: %s", exc)

        # --- FALLBACK: world position nearest-segment search ---
        if layer_data is None:
            return self.extract_hover_data(actor)

        if pick_pos is not None:
            result = self._find_nearest_segment(pick_pos, layer_data)
            if result is not None:
                return result

        # Fallback: try to identify the hole from the actor
        aid = id(actor)
        hole_id = self._actor_id_to_hole.get(aid, "")
        if hole_id:
            props = self._get_hole_basic_props(hole_id)
            return PickResult(
                hit=True,
                layer_name=f"Drillhole: {hole_id}",
                layer_type="drillhole",
                identifier=hole_id,
                properties=props,
            )

        # Merged actor without pick position — show basic info
        return PickResult(
            hit=True,
            layer_name="Drillholes",
            layer_type="drillhole",
            identifier="",
        )

    def invalidate_cache(self) -> None:
        self._actor_id_to_hole.clear()
        self._is_merged = False
        self._merged_actor_id = None

    # -- internal --

    def _ensure_cache(self) -> bool:
        actors = getattr(self._renderer, "_drillhole_hole_actors", {})
        if not actors:
            self._actor_id_to_hole.clear()
            self._is_merged = False
            self._merged_actor_id = None
            return False

        if "_merged" in actors:
            self._is_merged = True
            self._merged_actor_id = id(actors["_merged"])
            self._actor_id_to_hole.clear()
        else:
            self._is_merged = False
            self._merged_actor_id = None
            self._actor_id_to_hole = {
                id(actor): hid
                for hid, actor in actors.items()
                if hid != "_merged" and hid != "_merged_hids"
            }

        return True

    @staticmethod
    def _point_to_segments_dist(pick_pt: np.ndarray,
                                seg_starts: np.ndarray,
                                seg_ends: np.ndarray) -> np.ndarray:
        """Vectorised perpendicular distance from *pick_pt* to each segment.

        For every segment (A, B) the closest point on the segment to *pick_pt*
        is  ``A + t * (B - A)``  where ``t = clamp(dot(P-A, B-A) / |B-A|^2, 0, 1)``.
        Returns an array of distances, one per segment.
        """
        ab = seg_ends - seg_starts                        # (N, 3)
        ap = pick_pt - seg_starts                         # (N, 3)
        ab_sq = np.einsum("ij,ij->i", ab, ab)             # |B-A|^2
        # Avoid division by zero for degenerate (zero-length) segments
        degen = ab_sq < 1e-12
        ab_sq_safe = np.where(degen, 1.0, ab_sq)
        t = np.einsum("ij,ij->i", ap, ab) / ab_sq_safe   # projection parameter
        t = np.clip(t, 0.0, 1.0)
        t[degen] = 0.0                                    # degenerate → use start point
        closest = seg_starts + t[:, np.newaxis] * ab      # (N, 3)
        diff = pick_pt - closest
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))

    def _find_nearest_segment(
        self, pick_pos: Tuple[float, float, float],
        layer_data: Dict[str, Any]
    ) -> Optional[PickResult]:
        """Find the nearest drillhole segment to the 3D pick position.

        Uses perpendicular point-to-line-segment distance (not midpoint)
        so the correct hole is identified even when holes are close together.
        """
        hole_polys = layer_data.get("hole_polys", {})
        if not hole_polys:
            return None

        pick_pt = np.array(pick_pos, dtype=float)
        best_dist = float("inf")
        best_hole_id = ""
        best_seg_idx = -1

        for hid, poly in hole_polys.items():
            if poly is None or not hasattr(poly, "points") or poly.n_points < 2:
                continue

            pts = np.asarray(poly.points)
            n_pts = len(pts)
            if n_pts < 2:
                continue

            # Vectorised perpendicular distance to every segment in this hole
            seg_starts = pts[:-1]           # (N-1, 3)
            seg_ends = pts[1:]              # (N-1, 3)
            dists = self._point_to_segments_dist(pick_pt, seg_starts, seg_ends)

            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])
            if min_dist < best_dist:
                best_dist = min_dist
                best_hole_id = hid
                best_seg_idx = min_idx

        if not best_hole_id or best_seg_idx < 0:
            return None

        # Build properties from the matched segment
        props: Dict[str, Any] = {"hole_id": best_hole_id}

        seg_lith = layer_data.get("hole_segment_lith", {}).get(best_hole_id, [])
        seg_assay = layer_data.get("hole_segment_assay", {}).get(best_hole_id, [])
        seg_from = layer_data.get("hole_segment_from_depth", {}).get(best_hole_id, [])
        seg_to = layer_data.get("hole_segment_to_depth", {}).get(best_hole_id, [])

        if 0 <= best_seg_idx < len(seg_lith) and seg_lith[best_seg_idx]:
            props["lithology"] = seg_lith[best_seg_idx]
        if 0 <= best_seg_idx < len(seg_assay):
            val = seg_assay[best_seg_idx]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                assay_field = layer_data.get("assay_field", "grade")
                props[assay_field] = round(float(val), 4)
        if 0 <= best_seg_idx < len(seg_from):
            props["from"] = round(float(seg_from[best_seg_idx]), 2)
        if 0 <= best_seg_idx < len(seg_to):
            props["to"] = round(float(seg_to[best_seg_idx]), 2)

        collar = layer_data.get("collar_coords", {}).get(best_hole_id)
        if collar:
            props["collar"] = f"({collar[0]:.1f}, {collar[1]:.1f}, {collar[2]:.1f})"

        # Store segment index for highlighting (internal, prefixed with _)
        props["_segment_idx"] = best_seg_idx

        logger.debug(
            f"DrillholePick: nearest segment {best_hole_id}[{best_seg_idx}] "
            f"dist={best_dist:.2f}"
        )

        return PickResult(
            hit=True,
            layer_name=f"Drillhole: {best_hole_id}",
            layer_type="drillhole",
            identifier=best_hole_id,
            properties=props,
            world_pos=pick_pos,
        )

    def _get_hole_basic_props(self, hole_id: str) -> Dict[str, Any]:
        """Get basic properties (collar) for a hole when segment data is unavailable."""
        props: Dict[str, Any] = {"hole_id": hole_id}
        layer_data = self._get_drillhole_layer_data()
        if layer_data:
            collar = layer_data.get("collar_coords", {}).get(hole_id)
            if collar:
                props["collar"] = f"({collar[0]:.1f}, {collar[1]:.1f}, {collar[2]:.1f})"
        return props

    def _get_drillhole_layer_data(self) -> Optional[Dict[str, Any]]:
        # Try active_layers first, then cache
        al = getattr(self._renderer, "active_layers", {})
        dh_layer = al.get("drillholes", {})
        data = dh_layer.get("data") if isinstance(dh_layer, dict) else None
        if isinstance(data, dict) and "hole_polys" in data:
            return data

        cache = getattr(self._renderer, "_drillhole_polylines_cache", None)
        if isinstance(cache, dict) and "hole_polys" in cache:
            return cache

        return None


# ---------------------------------------------------------------------------
# Surface / geology adapter
# ---------------------------------------------------------------------------

class SurfacePickAdapter:
    """Pick adapter for geology surfaces, solids, and generic meshes."""

    _SURFACE_TYPES = frozenset({
        "geology_surface", "geology_solid", "geology_contact",
        "geology_wireframe", "geology_contacts", "mesh", "surface",
        "pit", "pushback", "schedule",
    })

    def __init__(self, renderer: Any):
        self._renderer = renderer
        self._actor_id_to_layer: Dict[int, str] = {}

    def can_handle(self, actor: Any) -> bool:
        self._ensure_cache()
        return id(actor) in self._actor_id_to_layer

    def extract_hover_data(self, actor: Any) -> PickResult:
        name = self._actor_id_to_layer.get(id(actor), "Surface")
        return PickResult(
            hit=True, layer_name=name, layer_type="surface", identifier=name,
        )

    def extract_cell_data(self, actor: Any, cell_id: int,
                          pick_pos: Optional[Tuple[float, float, float]] = None) -> PickResult:
        name = self._actor_id_to_layer.get(id(actor), "Surface")
        layer_info = self._renderer.active_layers.get(name, {})
        data = layer_info.get("data")

        props: Dict[str, Any] = {}
        mesh = None
        if isinstance(data, dict):
            mesh = data.get("mesh", data)
        elif data is not None:
            mesh = data

        if mesh is not None and hasattr(mesh, "cell_data") and cell_id >= 0:
            for key in list(mesh.cell_data.keys())[:10]:
                # I1 FIX: Skip internal arrays in surface adapter too
                if BlockModelPickAdapter._is_internal_array(key):
                    continue
                arr = mesh.cell_data[key]
                if cell_id < len(arr):
                    val = arr[cell_id]
                    # I3 FIX: Skip multi-component arrays
                    if isinstance(val, np.ndarray) and val.ndim > 0:
                        continue
                    if isinstance(val, (np.floating, float)) and np.isnan(val):
                        continue
                    props[key] = val.item() if hasattr(val, "item") else val

        return PickResult(
            hit=True,
            layer_name=name,
            layer_type="surface",
            identifier=name,
            properties=props,
        )

    def invalidate_cache(self) -> None:
        self._actor_id_to_layer.clear()

    def _ensure_cache(self) -> bool:
        self._actor_id_to_layer.clear()
        for name, info in getattr(self._renderer, "active_layers", {}).items():
            ltype = info.get("type", "")
            if ltype in self._SURFACE_TYPES:
                layer_actor = info.get("actor")
                if layer_actor is not None:
                    self._actor_id_to_layer[id(layer_actor)] = name

        return bool(self._actor_id_to_layer)


# ---------------------------------------------------------------------------
# Tooltip overlay widget
# ---------------------------------------------------------------------------

class TooltipOverlay(QFrame):
    """Custom tooltip QFrame positioned in viewport screen space."""

    OFFSET_X = 16
    OFFSET_Y = 16
    MAX_WIDTH = 320
    CORNER_RADIUS = 6

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMaximumWidth(self.MAX_WIDTH)
        self._prop_labels: List[QWidget] = []
        self._setup_ui()
        self._apply_theme()
        self.hide()

    # -- UI setup --

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self._header = QLabel()
        self._header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._header.setWordWrap(True)
        layout.addWidget(self._header)

        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setFixedHeight(1)
        layout.addWidget(self._separator)

        self._props_layout = QGridLayout()
        self._props_layout.setSpacing(2)
        self._props_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._props_layout)

        self._footer = QLabel()
        self._footer.setFont(QFont("Segoe UI", 7))
        self._footer.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._footer.hide()
        layout.addWidget(self._footer)

    def _apply_theme(self) -> None:
        try:
            from .modern_styles import get_theme_colors

            c = get_theme_colors()
            self.setStyleSheet(
                f"TooltipOverlay {{"
                f"  background-color: {c.ELEVATED_BG};"
                f"  border: 1px solid {c.BORDER};"
                f"  border-radius: {self.CORNER_RADIUS}px;"
                f"  color: {c.TEXT_PRIMARY};"
                f"}}"
            )
            self._header.setStyleSheet(f"color: {c.ACCENT_PRIMARY}; background: transparent;")
            self._separator.setStyleSheet(f"background-color: {c.BORDER};")
            self._footer.setStyleSheet(f"color: {c.TEXT_SECONDARY}; background: transparent;")
            self._theme_colors = c
        except Exception:
            self._theme_colors = None

    # -- public API --

    def update_content(self, result: PickResult, is_locked: bool = False) -> None:
        header_text = result.layer_name
        if result.identifier:
            header_text = f"{result.layer_name}  [{result.identifier}]"
        self._header.setText(header_text)

        # Clear old property labels
        for w in self._prop_labels:
            self._props_layout.removeWidget(w)
            w.deleteLater()
        self._prop_labels.clear()

        row = 0
        max_rows = 8
        tc = self._theme_colors
        sec_color = tc.TEXT_SECONDARY if tc else "#888"

        for key, value in list(result.properties.items()):
            if key.startswith("_"):
                continue  # Skip internal keys like _segment_idx
            if row >= max_rows:
                break
            kl = QLabel(f"{key}:")
            kl.setFont(QFont("Segoe UI", 8))
            kl.setStyleSheet(f"color: {sec_color}; background: transparent;")

            if isinstance(value, float):
                vs = f"{value:.4f}"
            elif isinstance(value, int) and abs(value) > 999:
                vs = f"{value:,}"
            else:
                vs = str(value)

            vl = QLabel(vs)
            vl.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            vl.setStyleSheet("background: transparent;")

            self._props_layout.addWidget(kl, row, 0)
            self._props_layout.addWidget(vl, row, 1)
            self._prop_labels.extend([kl, vl])
            row += 1

        if is_locked:
            self._footer.setText("Click elsewhere or Esc to dismiss")
            self._footer.show()
        else:
            self._footer.hide()

        self.adjustSize()

    def position_near_screen(self, screen_cursor: QPoint) -> None:
        """Position tooltip near the cursor using screen coordinates."""
        tw, th = self.sizeHint().width(), self.sizeHint().height()

        screen = QApplication.screenAt(screen_cursor)
        if screen is not None:
            geom = screen.availableGeometry()
        else:
            geom = QApplication.primaryScreen().availableGeometry()

        x = screen_cursor.x() + self.OFFSET_X
        y = screen_cursor.y() + self.OFFSET_Y

        if x + tw > geom.right() - 4:
            x = screen_cursor.x() - tw - self.OFFSET_X
        if y + th > geom.bottom() - 4:
            y = screen_cursor.y() - th - self.OFFSET_Y

        x = max(geom.left() + 4, x)
        y = max(geom.top() + 4, y)

        self.move(x, y)

    def show_tooltip(self) -> None:
        self.show()
        self.raise_()

    def hide_tooltip(self) -> None:
        self.hide()

    def refresh_theme(self) -> None:
        self._apply_theme()


# ---------------------------------------------------------------------------
# Click inspector controller
# ---------------------------------------------------------------------------

class HoverInspectorController(QObject):
    """
    Click-to-inspect controller.

    Left-click on a block/drillhole/surface to show a tooltip with
    cell-level properties.  Click elsewhere or press Esc to dismiss.

    Bypasses LOD cell-level gates for clicks (vtkCellPicker on 500k
    cells takes ~50-150ms, acceptable for user-initiated clicks).
    """

    inspect_data_ready = pyqtSignal(object)  # PickResult

    def __init__(self, viewer_widget: QWidget, parent: Optional[QObject] = None):
        super().__init__(parent or viewer_widget)

        self._viewer = viewer_widget
        self._renderer = getattr(viewer_widget, "renderer", None)
        self._picking_ctrl = get_picking_controller()

        self._inspecting: bool = False
        self._last_pick: Optional[PickResult] = None

        # Overlay
        self._overlay = TooltipOverlay(viewer_widget)

        # Pick adapters (order: first match wins)
        self._adapters: List[Any] = []
        if self._renderer is not None:
            self._adapters = [
                BlockModelPickAdapter(self._renderer),
                DrillholePickAdapter(self._renderer),
                SurfacePickAdapter(self._renderer),
            ]

        logger.info("ClickInspectorController initialized")

    # ====================================================================
    # Public API (called from ViewerWidget)
    # ====================================================================

    @property
    def inspecting(self) -> bool:
        return self._inspecting

    def on_click(self, x: int, y: int) -> Optional[PickResult]:
        """Handle click: cell-level pick, show tooltip, return result.

        Returns PickResult so the caller can emit signals and highlight.
        """
        if not self._picking_ctrl.click_allowed:
            return None

        # Dismiss any existing tooltip first
        if self._inspecting:
            self._dismiss()

        result = self._do_click_pick(x, y)

        if result.hit:
            self._inspecting = True
            self._last_pick = result
            self._overlay.update_content(result, is_locked=True)
            self._overlay.position_near_screen(QCursor.pos())
            self._overlay.show_tooltip()
            self.inspect_data_ready.emit(result)
            logger.info(
                f"ClickInspector: picked {result.layer_name} "
                f"[{result.identifier}] ({len(result.properties)} props)"
            )
            return result
        else:
            self._dismiss()
            logger.debug("ClickInspector: click miss")
            return None

    def on_escape(self) -> None:
        """Dismiss tooltip on Esc."""
        if self._inspecting:
            self._dismiss()

    def on_data_changed(self) -> None:
        """Invalidate adapter caches and dismiss tooltip."""
        for adapter in self._adapters:
            adapter.invalidate_cache()
        if self._inspecting:
            self._dismiss()

    # ====================================================================
    # Internal
    # ====================================================================

    def _dismiss(self) -> None:
        """Clear inspection state and hide tooltip."""
        self._overlay.hide_tooltip()
        self._inspecting = False
        self._last_pick = None

    def _do_click_pick(self, x: int, y: int) -> PickResult:
        """Cell-level pick that bypasses LOD restrictions.

        Always tries cell picker first (acceptable latency for clicks),
        falls back to prop picker only if cell pick misses.
        """
        plotter = getattr(self._viewer, "plotter", None)
        if plotter is None:
            return PickResult.miss()
        vtk_renderer = plotter.renderer
        if vtk_renderer is None:
            return PickResult.miss()

        # FIX: Apply high-DPI scaling. Qt reports mouse coords in logical
        # pixels but VTK expects physical pixels.  On 150% scaling,
        # logical (200,300) → physical (300,450).  Without this, picks
        # are offset from the visual position.
        dpr = 1.0
        try:
            from .dpi_utils import _get_dpi_ratio
            dpr = _get_dpi_ratio()
        except Exception:
            try:
                dpr = self._viewer.devicePixelRatioF()
            except Exception:
                pass
        vtk_x = int(x * dpr)
        vtk_y = int((self._viewer.height() - y - 1) * dpr)

        t0 = time.perf_counter()

        # ALWAYS try cell-level first for clicks (bypass LOD gate)
        if self._picking_ctrl.click_cell_level_allowed:
            result = self._do_cell_pick_vtk(vtk_x, vtk_y, vtk_renderer)
            if result.hit:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(f"ClickInspector: cell pick in {elapsed:.1f}ms (dpr={dpr:.2f})")
                return result

        # Fall back to actor-level
        result = self._do_prop_pick(vtk_x, vtk_y, vtk_renderer)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"ClickInspector: prop pick in {elapsed:.1f}ms (hit={result.hit})")
        return result

    def _do_cell_pick_vtk(self, vtk_x: int, vtk_y: int, vtk_renderer: Any) -> PickResult:
        """Cell-level pick at VTK coordinates.

        Captures the 3D pick position and passes it to adapters.
        This is critical for drillholes where the tube mesh cell_id
        doesn't correspond to original line segment indices.
        """
        cell_picker = self._picking_ctrl.get_cell_picker()
        if cell_picker is None:
            return PickResult.miss()

        # E2 FIX: Guard VTK picker calls against C++ exceptions (corrupt
        # renderer state, invalid viewport coordinates, etc.) so the UI
        # never crashes from a failed pick — it simply returns a miss.
        try:
            cell_picker.Pick(vtk_x, vtk_y, 0, vtk_renderer)
            actor = cell_picker.GetActor()
            cell_id = cell_picker.GetCellId()
        except Exception as exc:
            logger.debug("VTK cell pick failed: %s", exc)
            return PickResult.miss()

        if actor is None or cell_id < 0:
            return PickResult.miss()

        # Get the 3D world position of the pick
        pick_pos = cell_picker.GetPickPosition()  # (x, y, z)

        for adapter in self._adapters:
            if adapter.can_handle(actor):
                return adapter.extract_cell_data(actor, cell_id, pick_pos=pick_pos)

        return PickResult.miss()

    def _do_prop_pick(self, vtk_x: int, vtk_y: int, vtk_renderer: Any) -> PickResult:
        """Actor-level pick (fast, O(actors)).

        When the prop picker hits, also retrieves the 3D pick position
        and tries cell-level extraction so drillhole segment info is
        available even when the cell picker missed (e.g. at extreme zoom).
        """
        prop_picker = self._picking_ctrl.get_prop_picker()
        if prop_picker is None:
            return PickResult.miss()

        try:
            prop_picker.Pick(vtk_x, vtk_y, 0, vtk_renderer)
            actor = prop_picker.GetActor()
        except Exception as exc:
            logger.debug("VTK prop pick failed: %s", exc)
            return PickResult.miss()

        if actor is None:
            return PickResult.miss()

        # Try cell-level extraction with pick position (works for drillholes)
        pick_pos = prop_picker.GetPickPosition()
        for adapter in self._adapters:
            if adapter.can_handle(actor):
                try:
                    result = adapter.extract_cell_data(actor, -1, pick_pos=pick_pos)
                    if result.hit:
                        return result
                except Exception:
                    pass
                return adapter.extract_hover_data(actor)

        return PickResult.miss()
