"""
drillhole_renderer.py — Drillhole layer coordination and actor management.

Extracted from render_orchestrator.py as part of the renderer/ package refactor.

MIGRATION NOTE: DrillholeRenderer currently holds a reference to the parent
Renderer orchestrator (self._renderer) for backward-compatible access to
shared state.  Over time, explicit dependency injection will replace this pattern.

NOTE: DrillholeGPURenderer (drillhole_gpu_renderer.py) handles GPU-accelerated
rendering internally. DrillholeRenderer is a coordination layer only.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Callable, Tuple

import numpy as np
import pyvista as pv

from ....drillholes.datamodel import DrillholeDatabase
from ....drillholes.drillhole_layer import build_drillhole_polylines

logger = logging.getLogger(__name__)


class DrillholeRenderer:
    """
    Coordinates drillhole layer management and actor lifecycle.

    Responsibilities:
    - Drillhole layer addition and removal
    - Individual and batch visibility toggling
    - Drillhole radius updates (in-place and full rebuild)
    - Label visibility
    - Legend metadata retrieval

    NOT responsible for:
    - GPU rendering (handled by DrillholeGPURenderer)
    - Calling plotter.render() (only orchestrator controls render timing,
      except where caller logic requires immediate feedback)
    - Camera positioning
    """

    def __init__(self, renderer: Any) -> None:
        """
        Args:
            renderer: Reference to the parent Renderer orchestrator.
                      Used for backward-compatible access to shared state.
        """
        self._renderer = renderer

    # ------------------------------------------------------------------
    # Spline-tube builder (Leapfrog-quality smooth tubes)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spline_tube(
        poly: pv.PolyData,
        radius: float,
        n_sides: int,
        segment_scalars: Optional[List[float]] = None,
        scalar_name: str = "assay",
    ) -> Optional[pv.PolyData]:
        """Build a spline-smoothed tube with per-interval colour.

        One continuous tube from collar to TD.  Colours are assigned via
        ``point_data`` using the ring structure of the tube mesh: each
        ring of ``n_sides`` points at the same spline position gets the
        grade of the drillhole interval it falls inside.

        Because all points within one interval share the same scalar
        value, colour is uniform within each interval with a minimal
        (sub-ring) transition at boundaries.

        Args:
            poly: PolyData polyline from desurvey.
            radius: Tube radius in world units.
            n_sides: Number of tube facets (8-16).
            segment_scalars: Per-segment scalar values (one per interval).
            scalar_name: Name for the ``point_data`` array on the tube.

        Returns:
            Tube PolyData with ``point_data[scalar_name]`` or *None*.
        """
        points = poly.points
        if len(points) < 2:
            return None

        # Spline through desurvey points (3× densification, min 6 pts)
        try:
            n_spline = max(len(points) * 3, 6)
            spline = pv.Spline(points, n_points=n_spline)
        except Exception:
            spline = poly
            n_spline = len(poly.points)

        tube = spline.tube(radius=radius, capping=False, n_sides=n_sides)
        if tube.n_cells < 1:
            return None

        # ── Point-based scalars (ring-index mapping) ─────────────────
        if segment_scalars is not None and len(segment_scalars) > 0:
            vals = np.asarray(segment_scalars, dtype=np.float64)
            n_intervals = len(vals)
            n_tube_pts = tube.n_points
            n_spline_pts = len(spline.points)

            # Interval boundaries as fraction of total hole length
            diffs = np.diff(points, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_len = cum_len[-1] if cum_len[-1] > 1e-12 else 1.0
            boundaries = cum_len / total_len  # len = n_points

            # Tube points: n_sides points per ring, n_spline_pts rings.
            # point i belongs to ring (i // n_sides).
            ring_idx = np.arange(n_tube_pts) // n_sides
            t_pts = ring_idx.astype(np.float64) / max(1, n_spline_pts - 1)

            # Map each point to its drillhole interval
            seg_idx = np.searchsorted(boundaries[1:], t_pts, side='right')
            seg_idx = np.clip(seg_idx, 0, n_intervals - 1)

            tube.point_data[scalar_name] = vals[seg_idx].astype(np.float32)

        return tube

    def add_drillhole_layer(
        self,
        database: DrillholeDatabase,
        composite_df=None,
        radius: float = 1.0,
        color_mode: str = "Lithology",
        assay_field: Optional[str] = None,
        visible_holes: Optional[Set[str]] = None,
        legend_title: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        lith_filter: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add drillholes as a layer to the renderer.
        Uses individual hole actors for instant visibility toggling.
        
        Args:
            database: DrillholeDatabase containing collars, surveys, assays, lithology
            composite_df: Optional DataFrame with composite data
            radius: Tube radius for drillholes
            color_mode: "Lithology" or "Assay"
            visible_holes: Set of hole IDs to show (None = show all)
            lith_filter: Optional list of lithology codes to show (empty list or None = show all)
        """
        if self._renderer.plotter is None:
            logger.warning("Cannot add drillhole layer: plotter not initialized")
            return
        
        try:
            logger.debug(
                "[DRILLHOLE DEBUG] add_drillhole_layer start: radius=%s, color_mode=%s",
                radius,
                color_mode,
            )
            def _progress(fraction: float, message: str) -> None:
                if progress_callback is None:
                    return
                try:
                    progress_callback(max(0.0, min(1.0, fraction)), message)
                except Exception as e:
                    # DR-011 fix: Log exception instead of silently swallowing
                    logger.debug(f"Progress callback failed: {e}")
            _progress(0.02, "Preparing drillhole polylines")
            # Remove existing drillhole layer if present
            self.remove_drillhole_layer()
            
            # Build polylines using shared helper (cache this for radius updates)
            # CRITICAL: Pass registry for persistent interval IDs (GPU picking stability)
            registry = getattr(self._renderer, '_registry', None)
            result = build_drillhole_polylines(database, composite_df, assay_field_name=assay_field, registry=registry)
            hole_polys = result["hole_polys"]
            hole_segment_lith = result["hole_segment_lith"]
            hole_segment_assay = result["hole_segment_assay"]
            lith_colors = result["lith_colors"]
            lith_to_index = result["lith_to_index"]
            assay_field = result["assay_field"]
            assay_min = result["assay_min"]
            assay_max = result["assay_max"]
            hole_ids = result["hole_ids"]
            collar_coords = result["collar_coords"]
            logger.debug(
                "[DRILLHOLE DEBUG] Polylines built: holes=%d, collars=%d",
                len(hole_ids),
                len(collar_coords),
            )
            
            # ================================================================
            # CRITICAL FIX: Apply coordinate shift to drillhole polylines
            # ================================================================
            # Drillholes must use the SAME coordinate shift as geological surfaces
            # to render together in the same scene. Without this, drillholes remain
            # at UTM coordinates (e.g., 500,000m) while geology is shifted to ~(0,0,0),
            # causing both to disappear due to camera clipping range issues.
            # ================================================================
            all_points = []
            for poly in hole_polys.values():
                if poly.n_points > 0:
                    all_points.append(poly.points)
            
            if all_points:
                all_points_stacked = np.vstack(all_points)
                # Apply the SAME coordinate shift used by geological surfaces
                # This locks _global_shift if not already set (first dataset authority)
                shifted_origin = self._renderer._to_local_precision(all_points_stacked[:1])  # Lock shift
                
                # Now transform ALL polyline points to local coordinates
                for hid, poly in hole_polys.items():
                    if poly.n_points > 0:
                        local_points = self._renderer._to_local_precision(poly.points.copy())
                        poly.points = local_points
                
                # Also transform collar coordinates to local system
                if self._renderer._global_shift is not None:
                    for hid in collar_coords:
                        cx, cy, cz = collar_coords[hid]
                        collar_coords[hid] = (
                            cx - self._renderer._global_shift[0],
                            cy - self._renderer._global_shift[1],
                            cz - self._renderer._global_shift[2]
                        )
                    logger.info(f"[DRILLHOLE RENDER] Applied coordinate shift to {len(collar_coords)} collars")
                
                # Log shifted bounds for verification
                dh_x_min, dh_x_max = float(all_points_stacked[:, 0].min()), float(all_points_stacked[:, 0].max())
                dh_y_min, dh_y_max = float(all_points_stacked[:, 1].min()), float(all_points_stacked[:, 1].max())
                dh_z_min, dh_z_max = float(all_points_stacked[:, 2].min()), float(all_points_stacked[:, 2].max())
                logger.info(
                    f"[DRILLHOLE RENDER] Original drillhole bounds (world): X=[{dh_x_min:.2f}, {dh_x_max:.2f}], "
                    f"Y=[{dh_y_min:.2f}, {dh_y_max:.2f}], Z=[{dh_z_min:.2f}, {dh_z_max:.2f}]"
                )
                if self._renderer._global_shift is not None:
                    logger.info(
                        f"[DRILLHOLE RENDER] Applied global shift: [{self._renderer._global_shift[0]:.2f}, "
                        f"{self._renderer._global_shift[1]:.2f}, {self._renderer._global_shift[2]:.2f}]"
                    )
            
            _progress(0.15, "Caching drillhole data")
            
            # Cache polylines data for radius updates
            # Store color_mode if provided - colors will be applied after loading
            assay_p98 = result.get("assay_p98", assay_max)
            self._renderer._drillhole_polylines_cache = {
                "hole_polys": hole_polys,
                "hole_segment_lith": hole_segment_lith,
                "hole_segment_assay": hole_segment_assay,
                "lith_colors": lith_colors,
                "lith_to_index": lith_to_index,
                "assay_field": assay_field,
                "assay_min": assay_min,
                "assay_max": assay_max,
                "assay_p98": assay_p98,
                "database": database,
                "composite_df": composite_df,
                "color_mode": color_mode,  # Store color_mode for later color application
                "collar_coords": collar_coords,
                "radius": radius,
            }
            
            # Determine which holes to show
            if visible_holes is None:
                visible_holes = set(hole_ids)
            
            # Apply lithology filter if specified
            # lith_filter is a list of lithology codes to show - if empty, show all
            if lith_filter and len(lith_filter) > 0:
                lith_filter_set = set(lith_filter)
                logger.info(f"Applying lithology filter: showing only {lith_filter}")
                
                # Filter out segments that don't match the lithology filter
                # This modifies hole_polys and hole_segment_lith in-place
                for hid in list(hole_polys.keys()):
                    segments = hole_segment_lith.get(hid, [])
                    assay_segments = hole_segment_assay.get(hid, [])
                    poly = hole_polys.get(hid)
                    
                    if poly is None or poly.n_points == 0:
                        continue
                    
                    # Find which segments match the filter
                    matching_indices = [i for i, lith in enumerate(segments) if lith in lith_filter_set]
                    
                    if not matching_indices:
                        # No matching segments - create empty polydata
                        hole_polys[hid] = pv.PolyData()
                        hole_segment_lith[hid] = []
                        hole_segment_assay[hid] = []
                        continue
                    
                    # Rebuild polylines with only matching segments
                    if len(matching_indices) < len(segments):
                        # Need to filter - rebuild the polyline from scratch
                        points = poly.points
                        original_lines = poly.lines
                        
                        # Parse original line connectivity
                        new_points = []
                        new_lines = []
                        new_lith = []
                        new_assay = []
                        point_map = {}  # old_idx -> new_idx
                        
                        line_idx = 0
                        seg_idx = 0
                        while line_idx < len(original_lines):
                            n_pts = original_lines[line_idx]
                            if seg_idx in matching_indices:
                                # Include this segment
                                line_pts = []
                                for j in range(1, n_pts + 1):
                                    old_pt_idx = original_lines[line_idx + j]
                                    if old_pt_idx not in point_map:
                                        point_map[old_pt_idx] = len(new_points)
                                        new_points.append(points[old_pt_idx])
                                    line_pts.append(point_map[old_pt_idx])
                                new_lines.extend([n_pts] + line_pts)
                                new_lith.append(segments[seg_idx])
                                new_assay.append(assay_segments[seg_idx] if seg_idx < len(assay_segments) else np.nan)
                            
                            line_idx += n_pts + 1
                            seg_idx += 1
                        
                        if new_points and new_lines:
                            new_poly = pv.PolyData(np.array(new_points, dtype=float))
                            new_poly.lines = np.array(new_lines, dtype=np.int64)
                            hole_polys[hid] = new_poly
                            hole_segment_lith[hid] = new_lith
                            hole_segment_assay[hid] = new_assay
                        else:
                            hole_polys[hid] = pv.PolyData()
                            hole_segment_lith[hid] = []
                            hole_segment_assay[hid] = []
                
                # Update cache with filtered data
                self._renderer._drillhole_polylines_cache["hole_polys"] = hole_polys
                self._renderer._drillhole_polylines_cache["hole_segment_lith"] = hole_segment_lith
                self._renderer._drillhole_polylines_cache["hole_segment_assay"] = hole_segment_assay
                
                # CRITICAL: Rebuild lith_to_index and lith_colors to match filtered lithologies
                # Otherwise property panel will show all lithologies, not just filtered ones
                filtered_unique_codes = sorted({code for codes in hole_segment_lith.values() for code in codes if code})
                
                if filtered_unique_codes:
                    # Rebuild lith_to_index with only filtered codes
                    filtered_lith_to_index = {code: idx for idx, code in enumerate(filtered_unique_codes)}
                    
                    # Rebuild lith_colors - preserve original colors for codes that remain
                    filtered_lith_colors = {}
                    for code in filtered_unique_codes:
                        if code in lith_colors:
                            filtered_lith_colors[code] = lith_colors[code]
                        else:
                            # Shouldn't happen, but provide fallback color just in case
                            filtered_lith_colors[code] = "#808080"  # Gray
                    
                    # Update cache and local variables with filtered mappings
                    lith_to_index = filtered_lith_to_index
                    lith_colors = filtered_lith_colors
                    self._renderer._drillhole_polylines_cache["lith_to_index"] = lith_to_index
                    self._renderer._drillhole_polylines_cache["lith_colors"] = lith_colors
                    
                    logger.info(f"Rebuilt lithology mappings after filter: {len(filtered_unique_codes)} codes remaining")
                else:
                    logger.warning("No lithology codes remaining after filter - using empty mappings")
                    lith_to_index = {}
                    lith_colors = {}
                    self._renderer._drillhole_polylines_cache["lith_to_index"] = lith_to_index
                    self._renderer._drillhole_polylines_cache["lith_colors"] = lith_colors
            
            # OPTION: Use GPU renderer for large datasets or if explicitly enabled
            # Auto-enable GPU renderer for datasets with >1000 holes or >10k intervals
            total_intervals = sum(len(segments) for segments in hole_segment_lith.values())
            # GPU renderer is currently experimental - only use if explicitly enabled
            # or for very large datasets where standard renderer would be too slow
            use_gpu = self._renderer._use_gpu_drillholes and (len(hole_ids) > 5000 or total_intervals > 100000)
            
            if use_gpu:
                try:
                    from ...drillhole_gpu_renderer import (
                        DrillholeGPURenderer,
                        create_intervals_from_polyline_data,
                        get_drillhole_event_bus,
                        RenderQuality,
                    )
                    
                    logger.info(f"Using GPU renderer for {len(hole_ids)} holes ({total_intervals} intervals)")
                    
                    # Store original radius for transform scaling
                    self._renderer._drillhole_original_radius = radius
                    
                    # Convert to GPU intervals
                    intervals = create_intervals_from_polyline_data(result, radius=radius)
                    
                    # Create GPU renderer
                    self._renderer._gpu_drillhole_renderer = DrillholeGPURenderer(
                        self._renderer.plotter,
                        quality=RenderQuality.HIGH,
                        enable_gpu_picking=True
                    )
                    
                    # Load intervals
                    self._renderer._gpu_drillhole_renderer.load_intervals(intervals)
                    
                    # Apply colors if color_mode is provided
                    if color_mode and color_mode != "None":
                        logger.info(f"Applying colors to GPU drillholes with color_mode={color_mode}")
                        try:
                            # Determine property name and colormap based on color_mode
                            if color_mode == "Lithology":
                                property_name = "lithology"
                                colormap = "tab10"
                            else:  # Assay
                                property_name = "assay"  # GPU renderer uses "assay" as property name
                                colormap = "turbo"
                            
                            # Render with colormap first
                            self._renderer._gpu_drillhole_renderer.render(colormap=colormap, show_collars=True)
                            
                            # Update color property if method exists
                            if hasattr(self._renderer._gpu_drillhole_renderer, 'update_color_property'):
                                self._renderer._gpu_drillhole_renderer.update_color_property(property_name=property_name)
                            
                            # Update colormap if method exists
                            if hasattr(self._renderer._gpu_drillhole_renderer, 'update_colormap'):
                                self._renderer._gpu_drillhole_renderer.update_colormap(colormap=colormap)
                                
                        except Exception as e:
                            logger.warning(f"Failed to apply colors to GPU drillholes: {e}")
                            # Render without colormap as fallback
                            self._renderer._gpu_drillhole_renderer.render(colormap="viridis", show_collars=True)
                    else:
                        logger.info("Loading GPU drillholes without color assignment - colors will be applied when property is selected")
                        # Render without colormap - use default uniform color
                        self._renderer._gpu_drillhole_renderer.render(colormap="viridis", show_collars=True)
                    
                    # Set up mouse event handlers for hover and click
                    self._renderer._setup_drillhole_interaction()
                    
                    # Connect to event bus for selection/hover feedback
                    event_bus = get_drillhole_event_bus()
                    event_bus.intervalSelected.connect(self._renderer._on_drillhole_interval_selected)
                    event_bus.intervalHovered.connect(self._renderer._on_drillhole_interval_hovered)
                    
                    # Store layer data with color_mode
                    layer_data = {
                        "database": database,
                        "composite_df": composite_df,
                        "radius": radius,
                        "color_mode": color_mode,  # Store color_mode
                        "hole_polys": hole_polys,
                        "hole_segment_lith": hole_segment_lith,
                        "hole_segment_assay": hole_segment_assay,
                        "lith_colors": lith_colors,
                        "lith_to_index": lith_to_index,
                        "assay_field": assay_field,
                        "assay_min": assay_min,
                        "assay_max": assay_max,
                        "hole_ids": hole_ids,
                        "collar_coords": collar_coords,
                        "visible_holes": visible_holes,
                        "gpu_renderer": self._renderer._gpu_drillhole_renderer,
                    }
                    
                    # Register GPU renderer actors with Visual Density Controller
                    if self._renderer.visual_density_controller is not None and self._renderer._gpu_drillhole_renderer._main_actor:
                        try:
                            # For GPU renderer, we have one main actor and collar actors
                            gpu_drillhole_actors = {"gpu_main": self._renderer._gpu_drillhole_renderer._main_actor}
                            gpu_collar_actors = getattr(self._renderer._gpu_drillhole_renderer, '_collar_actors', {})

                            self._renderer.visual_density_controller.register_actors(
                                drillhole_actors=gpu_drillhole_actors,
                                collar_actors=gpu_collar_actors,
                                label_actors=[],
                            )
                            logger.debug("Registered GPU drillhole actors with VisualDensityController")
                        except Exception as e:
                            logger.warning(f"Failed to register GPU actors with VisualDensityController: {e}")

                    # Use first actor from GPU renderer
                    if self._renderer._gpu_drillhole_renderer._main_actor:
                        self._renderer.add_layer("drillholes", self._renderer._gpu_drillhole_renderer._main_actor,
                                     data=layer_data, layer_type="drillhole", opacity=1.0)

                    logger.info("GPU drillhole renderer initialized with hover/click support")
                    # Activate resize debounce for GPU drillhole scenes
                    if hasattr(self._renderer, '_resize_debounce') and self._renderer._resize_debounce is not None:
                        self._renderer._resize_debounce.activate()
                        logger.info("[RESIZE DEBOUNCE] Activated for GPU drillhole scene")
                    _progress(1.0, "Drillholes ready (GPU accelerated)")
                    return layer_data
                    
                except Exception as e:
                    logger.warning(f"GPU renderer failed, falling back to standard renderer: {e}")
                    self._renderer._gpu_drillhole_renderer = None
                    # Continue with standard renderer below
            logger.debug(
                "[DRILLHOLE DEBUG] visible holes count = %d (total=%d)",
                len(visible_holes),
                len(hole_ids),
            )
            total_holes = len(hole_ids)
            
            # Store original radius for transform scaling
            self._renderer._drillhole_original_radius = radius
            
            # Apply colors if color_mode is provided, otherwise load without colors
            if color_mode and color_mode != "None":
                logger.info(f"Loading drillholes with color_mode={color_mode} - colors will be applied after loading")
                use_colors_during_loading = True
                loading_color_mode = color_mode
            else:
                logger.info("Loading drillholes without color assignment - colors will be applied when property is selected")
                use_colors_during_loading = False
                loading_color_mode = None
            scalar_name = None  # Will be set when colors are applied
            
            # ULTRA-OPTIMIZED: Merge polylines first, then apply ONE tube filter
            # This is 10-100x faster than creating individual tubes
            self._renderer._drillhole_hole_actors = {}
            all_bounds = []
            
            # Adaptive quality based on dataset size
            # Minimum 12 sides ensures tubes look cylindrical, not polygonal
            n_sides = 16  # Default high quality
            if total_holes > 500:
                n_sides = 12  # Medium-high quality for large datasets
            elif total_holes > 200:
                n_sides = 14  # High quality
            
            _progress(0.10, f"Preparing {total_holes} drillholes...")
            
            # Phase 1: Collect polylines WITH hole IDs (preserve mapping)
            # Colors will be assigned later when user selects a property
            polylines_with_ids = []  # List of (poly, hid) tuples to preserve mapping
            all_scalars = []
            
            for hid in hole_ids:
                poly = hole_polys.get(hid)
                if poly is None or poly.n_cells < 1:
                    continue
                
                # Do NOT assign any scalar data during loading
                # Scalar data will be assigned when user explicitly selects a property
                
                polylines_with_ids.append((poly, hid))
                all_bounds.append(poly.bounds)
            
            if not polylines_with_ids:
                logger.warning("No drillhole polylines to render")
                _progress(1.0, "No drillhole polylines to render")
                return
            
            # Phase 2: Build spline-smoothed tubes (Leapfrog-quality)
            _progress(0.20, f"Building {len(polylines_with_ids)} drillhole tubes...")
            logger.info(f"Building spline tubes for {len(polylines_with_ids)} holes...")

            self._renderer._drillhole_hole_actors = {}
            tubes_with_ids = []

            for idx, (poly, hid) in enumerate(polylines_with_ids):
                # Build spline-smoothed tube with vertex-interpolated scalars
                tube = self._build_spline_tube(poly, radius, n_sides)
                if tube is None or tube.n_cells < 1:
                    continue

                tubes_with_ids.append((tube, hid))

                # Progress update every 50 holes (throttled to avoid signal flooding)
                if (idx + 1) % 50 == 0:
                    frac = 0.20 + 0.50 * ((idx + 1) / len(polylines_with_ids))
                    _progress(frac, f"Building tubes: {idx + 1}/{len(polylines_with_ids)}")
                    # Force UI repaint using safer processEvents
                    from PyQt6.QtCore import QEventLoop
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
            if not tubes_with_ids:
                logger.warning("No drillhole tubes to render")
                _progress(1.0, "No drillhole tubes to render")
                return
            
            # Phase 3: Create individual actors (enables individual visibility control)
            _progress(0.70, f"Creating {len(tubes_with_ids)} drillhole actors...")
            logger.info(f"Creating actors for {len(tubes_with_ids)} holes...")
            
            for idx, (tube, hid) in enumerate(tubes_with_ids):
                # Create actor with PBR materials (Leapfrog-quality shading)
                # Colors will be applied when user selects a property
                actor = self._renderer.plotter.add_mesh(
                    tube,
                    color='lightgray',
                    show_scalar_bar=False,
                    reset_camera=False,
                    smooth_shading=True,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.5,
                    nan_color="gray",
                    pickable=True,
                )
                
                # Set visibility based on visible_holes set
                if hid in visible_holes:
                    actor.VisibilityOn()
                else:
                    actor.VisibilityOff()
                
                self._renderer._drillhole_hole_actors[hid] = actor
                all_bounds.append(tube.bounds)
                
                # Progress update every 50 actors (throttled to avoid signal flooding)
                if (idx + 1) % 50 == 0:
                    frac = 0.70 + 0.10 * ((idx + 1) / len(tubes_with_ids))
                    _progress(frac, f"Creating actors: {idx + 1}/{len(tubes_with_ids)}")
                    # Force UI repaint using safer processEvents
                    from PyQt6.QtCore import QEventLoop
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
            if not actor:
                logger.warning("Failed to create drillhole actor")
                _progress(1.0, "Failed to create drillhole actor")
                return
            
            logger.info(f"Created {len(self._renderer._drillhole_hole_actors)} individual drillhole actors")
            
            # Add collar markers — glyphed spheres (Leapfrog-style)
            # Single glyph call for all collars = single VTK actor
            _progress(0.85, "Adding collar markers")
            self._renderer._drillhole_collar_actors = {}
            collar_radius = radius * 0.6  # Slightly larger than tube

            visible_collar_hids = []
            collar_pts = []
            for hid in hole_ids:
                if hid not in visible_holes:
                    continue
                collar = collar_coords.get(hid)
                if collar is None:
                    continue
                visible_collar_hids.append(hid)
                collar_pts.append(collar)

            if collar_pts:
                collar_cloud = pv.PolyData(np.array(collar_pts, dtype=np.float64))
                collar_glyphs = collar_cloud.glyph(
                    geom=pv.Sphere(radius=collar_radius),
                    orient=False,
                    scale=False,
                )
                collar_actor = self._renderer.plotter.add_mesh(
                    collar_glyphs,
                    color="white",
                    smooth_shading=True,
                    pbr=True,
                    metallic=0.3,
                    roughness=0.4,
                    show_scalar_bar=False,
                    reset_camera=False,
                    pickable=True,
                    name="drillhole_collars_merged",
                )
                self._renderer._drillhole_collar_actors["_merged"] = collar_actor
                self._renderer._drillhole_collar_actors["_merged_hids"] = visible_collar_hids
                logger.info(f"Created glyphed collar spheres for {len(visible_collar_hids)} collars")

            # Register actors with Visual Density Controller for automatic LOD
            if self._renderer.visual_density_controller is not None:
                try:
                    self._renderer.visual_density_controller.register_actors(
                        drillhole_actors=self._renderer._drillhole_hole_actors,
                        collar_actors=self._renderer._drillhole_collar_actors,
                        label_actors=[],  # No label actors in standard renderer
                    )
                    logger.debug(f"Registered {len(self._renderer._drillhole_hole_actors)} drillhole actors with VisualDensityController")
                except Exception as e:
                    logger.warning(f"Failed to register actors with VisualDensityController: {e}")

            # Store layer data (using first actor as representative)
            first_actor = next(iter(self._renderer._drillhole_hole_actors.values()))
            layer_data = {
                "database": database,
                "composite_df": composite_df,
                "radius": radius,
                "color_mode": color_mode,
                "hole_polys": hole_polys,
                "hole_segment_lith": hole_segment_lith,
                "hole_segment_assay": hole_segment_assay,
                "lith_colors": lith_colors,
                "lith_to_index": lith_to_index,
                "assay_field": assay_field or result.get("assay_field"),
                "assay_min": assay_min,
                "assay_max": assay_max,
                "hole_ids": hole_ids,
                "collar_coords": collar_coords,
                "visible_holes": visible_holes,
            }
            
            self._renderer.add_layer("drillholes", first_actor, data=layer_data, layer_type="drillhole", opacity=1.0)
            
            # Register in scene_layers for global picking and state management
            # CRITICAL: Must register BEFORE triggering callback so state check finds the layer
            self._renderer.register_scene_layer("drillholes", first_actor, layer_data, "drillhole")
            
            # Trigger another callback now that scene_layers is populated
            # This ensures the app state transitions to RENDERED
            if self._renderer.layer_change_callback:
                self._renderer.layer_change_callback()
            
            # Explicitly ensure layer is visible
            if "drillholes" in self._renderer.active_layers:
                self._renderer.active_layers["drillholes"]['visible'] = True
            
            # Ensure all drillhole actors are visible (fix for visibility issue)
            # NOTE: After _update_drillhole_colors is called, individual actors may be replaced
            # with a single "_merged" actor. Handle both cases.
            if "_merged" in self._renderer._drillhole_hole_actors:
                # Merged actor mode (after color update)
                try:
                    self._renderer._drillhole_hole_actors["_merged"].VisibilityOn()
                    logger.debug("Enabled visibility for merged drillhole actor")
                except Exception as e:
                    logger.warning(f"Failed to enable visibility for merged drillhole actor: {e}")
            else:
                # Individual actor mode (before color update)
                for hid, actor in self._renderer._drillhole_hole_actors.items():
                    if hid in visible_holes:
                        try:
                            actor.VisibilityOn()
                        except Exception:
                            pass
            
            # Ensure collar actor is visible (merged mode)
            if "_merged" in self._renderer._drillhole_collar_actors:
                try:
                    self._renderer._drillhole_collar_actors["_merged"].VisibilityOn()
                except Exception:
                    pass
            else:
                # Legacy individual collar actors
                for hid, actor in self._renderer._drillhole_collar_actors.items():
                    if hid.startswith("_"):
                        continue  # Skip metadata keys
                    if hid in visible_holes:
                        try:
                            actor.VisibilityOn()
                        except Exception:
                            pass
            
            # Ensure the representative actor is visible (may be merged or individual)
            if first_actor is not None:
                try:
                    first_actor.VisibilityOn()
                except Exception:
                    pass
            
            # Force render to ensure visibility changes take effect
            if self._renderer.plotter is not None:
                try:
                    self._renderer.plotter.render()
                except Exception:
                    pass

            # Re-apply SSAO/EDL — add_mesh resets VTK render passes
            self._renderer.reapply_scene_effects()

            # Initialize state manager with drillhole data
            if self._renderer._drillhole_state_manager:
                try:
                    self._renderer._drillhole_state_manager.register_holes(list(hole_ids))
                    # Set visibility for all holes
                    for hid in hole_ids:
                        self._renderer._drillhole_state_manager.set_visibility(hid, hid in visible_holes)
                    self._renderer._drillhole_state_manager.set_color_property(
                        "lithology" if color_mode == "Lithology" else "assay"
                    )
                    self._renderer._drillhole_state_manager.set_colormap("tab10" if color_mode == "Lithology" else "turbo")
                    self._renderer._drillhole_state_manager.set_tube_radius(radius)
                except Exception as e:
                    logger.debug(f"State manager initialization failed: {e}")
            
            logger.debug(
                "[DRILLHOLE DEBUG] Added drillhole layer with %d actors; visible_holes=%d",
                len(self._renderer._drillhole_hole_actors),
                len(visible_holes),
            )
            
            # Set up click and hover for standard PyVista renderer
            self._renderer._setup_standard_drillhole_interaction(hole_polys, hole_segment_lith, hole_segment_assay, database)
            
            try:
                self._renderer._update_scene_bounds()
                logger.debug(
                    "[DRILLHOLE DEBUG] Scene bounds after drillhole update: %s",
                    self._renderer._fixed_scene_bounds,
                )
            except Exception as exc:
                logger.debug(f"Could not refresh scene bounds after drillhole load: {exc}")
            _progress(0.9, "Positioning camera for drillholes")
            
            # Reset camera to fit drillholes
            try:
                if all_bounds:
                    # Calculate combined bounds
                    min_x = min(b[0] for b in all_bounds)
                    max_x = max(b[1] for b in all_bounds)
                    min_y = min(b[2] for b in all_bounds)
                    max_y = max(b[3] for b in all_bounds)
                    min_z = min(b[4] for b in all_bounds)
                    max_z = max(b[5] for b in all_bounds)
                    
                    drill_bounds = (min_x, max_x, min_y, max_y, min_z, max_z)
                    if self._renderer._fixed_scene_bounds is None:
                        self._renderer._fixed_scene_bounds = drill_bounds
                    else:
                        fb = self._renderer._fixed_scene_bounds
                        self._renderer._fixed_scene_bounds = (
                            min(fb[0], drill_bounds[0]),
                            max(fb[1], drill_bounds[1]),
                            min(fb[2], drill_bounds[2]),
                            max(fb[3], drill_bounds[3]),
                            min(fb[4], drill_bounds[4]),
                            max(fb[5], drill_bounds[5]),
                        )
                    try:
                        self._renderer._maintain_clipping_range()
                        cam_clip = (
                            self._renderer.plotter.renderer.GetActiveCamera().GetClippingRange()
                            if self._renderer.plotter.renderer is not None
                            else None
                        )
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera clipping after drillhole bounds merge: %s",
                            cam_clip,
                        )
                    except Exception:
                        pass
                    
                    center = [
                        (min_x + max_x) / 2,
                        (min_y + max_y) / 2,
                        (min_z + max_z) / 2,
                    ]
                    size = max(max_x - min_x, max_y - min_y, max_z - min_z)
                    if size <= 0:
                        size = 1.0
                    
                    camera = self._renderer.plotter.renderer.GetActiveCamera()
                    if camera:
                        cam_distance = size * 1.5
                        new_pos = [
                            center[0] + cam_distance,
                            center[1] + cam_distance,
                            center[2] + cam_distance,
                        ]
                        camera.SetFocalPoint(center)
                        camera.SetPosition(new_pos)
                        camera.SetViewUp(0, 0, 1)
                        # Set clipping range directly from known scene geometry instead of
                        # ResetCameraClippingRange() which uses ALL renderer actors (including
                        # any overlay actors at wrong coordinates) and can produce a near clip
                        # larger than the camera-to-data distance, making everything invisible.
                        near = max(0.001, size * 0.001)
                        far = max(size * 100.0, cam_distance * 100.0)
                        camera.SetClippingRange(near, far)
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera positioned for drillholes: pos=%s focal=%s clip=(%s, %s)",
                            camera.GetPosition(),
                            camera.GetFocalPoint(),
                            near,
                            far,
                        )
                    else:
                        self._renderer.plotter.reset_camera(bounds=drill_bounds)
                else:
                    self._renderer.plotter.reset_camera()
            except Exception as e:
                logger.warning(f"Failed to position camera for drillholes: {e}")
                try:
                    self._renderer.plotter.reset_camera()
                except Exception:
                    pass
            
            # Force render
            try:
                self._renderer.plotter.render()
                try:
                    cam = self._renderer.plotter.renderer.GetActiveCamera()
                    if cam:
                        logger.debug(
                            "[DRILLHOLE DEBUG] Camera after render: pos=%s focal=%s clip=%s",
                            cam.GetPosition(),
                            cam.GetFocalPoint(),
                            cam.GetClippingRange(),
                        )
                except Exception:
                    pass
                _progress(1.0, "Drillholes ready")
            except Exception as e:
                logger.warning(f"Failed to render after adding drillholes: {e}")
            
            logger.info(f"Added drillhole layer with {len(self._renderer._drillhole_hole_actors)} individual actors")

            # Activate resize debounce for drillhole scenes
            # Drillhole polylines + collar markers + potential label actors make
            # resize-triggered renders expensive enough to warrant debouncing
            if hasattr(self._renderer, '_resize_debounce') and self._renderer._resize_debounce is not None:
                self._renderer._resize_debounce.activate()
                logger.info("[RESIZE DEBOUNCE] Activated for drillhole scene")

            # Apply colors if color_mode was provided
            if color_mode and color_mode != "None" and len(self._renderer._drillhole_hole_actors) > 0:
                logger.info(f"Applying colors to drillholes with color_mode={color_mode}")
                try:
                    # Determine property name and colormap based on color_mode
                    if color_mode == "Lithology":
                        property_name = "lithology"
                        colormap = "tab10"
                        color_mode_param = "discrete"
                    else:  # Assay
                        property_name = assay_field or "assay"
                        colormap = "turbo"
                        color_mode_param = "continuous"
                    
                    # Apply colors using the update method
                    self._renderer._update_drillhole_colors(
                        property_name=property_name,
                        colormap=colormap,
                        color_mode=color_mode_param,
                        custom_colors=None
                    )
                    
                    # Update current property tracking
                    self._renderer.current_property = property_name
                    self._renderer.current_colormap = colormap
                    
                    # Create legend metadata with color information
                    # CRITICAL: Use lith_colors keys (which are updated by _update_drillhole_colors)
                    # instead of lith_to_index keys (which may not be updated)
                    cache = self._renderer._drillhole_polylines_cache
                    if color_mode == "Lithology":
                        lith_colors_from_cache = cache.get("lith_colors", {})
                        logger.info(f"[LEGEND CREATE] cache lith_colors has {len(lith_colors_from_cache)} entries: {list(lith_colors_from_cache.keys())[:5]}")
                        unique_liths = sorted(list(lith_colors_from_cache.keys()))
                        legend_metadata: Dict[str, Any] = {
                            "property": property_name,
                            "title": legend_title or "Drillholes",
                            "mode": "discrete",
                            "colormap": colormap,
                            "categories": unique_liths,
                            "category_colors": cache.get("lith_colors", {}),
                            "vmin": None,
                            "vmax": None,
                            "scalar_name": "lith_id",
                            "color_mode": color_mode,
                        }
                    else:
                        legend_metadata: Dict[str, Any] = {
                            "property": property_name,
                            "title": legend_title or "Drillholes",
                            "mode": "continuous",
                            "colormap": colormap,
                            "categories": None,
                            "category_colors": None,
                            "vmin": cache.get("assay_min", 0.0),
                            "vmax": cache.get("assay_max", 1.0),
                            "scalar_name": property_name,  # Use actual property name (Cu, Au, etc.) instead of generic "assay"
                            "color_mode": color_mode,
                        }
                    
                    # Store legend metadata
                    self._renderer._drillhole_legend_metadata = legend_metadata
                    
                    logger.info(f"Applied colors to drillholes: property={property_name}, colormap={colormap}")
                    
                    return legend_metadata
                    
                except Exception as e:
                    logger.warning(f"Failed to apply colors to drillholes: {e}", exc_info=True)
                    # Fall through to return empty legend metadata
            
            # No property assigned during loading - drillholes load without colors
            property_name = None  # No property selected initially
            self._renderer.current_property = None
            self._renderer.current_colormap = None
            
            # Skip legend creation during loading - legend will be created when user selects a property
            # Store empty legend metadata to indicate no colors are assigned
            legend_metadata: Dict[str, Any] = {
                "property": None,
                "title": legend_title or "Drillholes",
                "mode": None,
                "colormap": None,
                "categories": None,
                "category_colors": None,
                "vmin": None,
                "vmax": None,
                "scalar_name": None,
                "color_mode": color_mode if color_mode else None,
            }
            
            # Store legend metadata (empty - no colors assigned)
            self._renderer._drillhole_legend_metadata = legend_metadata
            
            # Do NOT update legend during loading - no colors are assigned
            # Legend will be updated when user explicitly selects a property
            logger.info("Skipping legend update during drillhole loading - no colors assigned")
            
            return legend_metadata
            
        except Exception as e:
            logger.error(f"Failed to add drillhole layer: {e}", exc_info=True)
            raise
    

    def remove_drillhole_layer(self) -> None:
        """Remove the drillhole layer if it exists."""
        # CRITICAL: Clear GPU renderer first if it exists (DR-001 fix)
        if hasattr(self._renderer, '_gpu_drillhole_renderer') and self._renderer._gpu_drillhole_renderer is not None:
            try:
                # Disconnect event bus signals to prevent stale handlers (DR-004 fix)
                from ...drillhole_gpu_renderer import get_drillhole_event_bus
                event_bus = get_drillhole_event_bus()
                try:
                    event_bus.intervalSelected.disconnect(self._renderer._on_drillhole_interval_selected)
                except (TypeError, RuntimeError):
                    pass  # Already disconnected or never connected
                try:
                    event_bus.intervalHovered.disconnect(self._renderer._on_drillhole_interval_hovered)
                except (TypeError, RuntimeError):
                    pass
                
                # Clear GPU renderer (stops timers, removes actors, clears state)
                self._renderer._gpu_drillhole_renderer.clear()
                logger.debug("[DRILLHOLE DEBUG] Cleared GPU drillhole renderer.")
            except Exception as e:
                logger.warning(f"Error clearing GPU drillhole renderer: {e}")
            finally:
                self._renderer._gpu_drillhole_renderer = None
        
        # Remove all individual hole actors (standard renderer)
        if self._renderer.plotter is not None:
            for hole_id, actor in self._renderer._drillhole_hole_actors.items():
                try:
                    self._renderer.plotter.remove_actor(actor)
                except Exception:
                    pass
        removed = len(self._renderer._drillhole_hole_actors)
        self._renderer._drillhole_hole_actors.clear()
        logger.debug("[DRILLHOLE DEBUG] Removed %d drillhole actors.", removed)
        
        # Remove collar marker actors
        if self._renderer.plotter is not None and hasattr(self._renderer, '_drillhole_collar_actors'):
            for key, collar_actor in self._renderer._drillhole_collar_actors.items():
                if key == "_merged_hids":
                    continue  # Skip metadata key (list of hids, not actor)
                try:
                    self._renderer.plotter.remove_actor(collar_actor)
                except Exception:
                    pass
            self._renderer._drillhole_collar_actors.clear()
            logger.debug("[DRILLHOLE DEBUG] Removed collar marker actors.")
        
        if "drillholes" in self._renderer.active_layers:
            self._renderer.clear_layer("drillholes")
        
        # Remove labels if they exist
        if hasattr(self._renderer, "_drillhole_label_actor") and self._renderer._drillhole_label_actor is not None:
            try:
                if self._renderer.plotter is not None:
                    self._renderer.plotter.remove_actor(self._renderer._drillhole_label_actor)
            except Exception as e:
                logger.debug(f"Could not remove drillhole label actor: {e}")
            self._renderer._drillhole_label_actor = None
        
        # Clear cache with proper memory release (Phase 2.2 fix)
        if self._renderer._drillhole_polylines_cache is not None:
            # Clear nested structures explicitly to help garbage collection
            for key in list(self._renderer._drillhole_polylines_cache.keys()):
                val = self._renderer._drillhole_polylines_cache[key]
                if isinstance(val, dict):
                    val.clear()
                elif isinstance(val, list):
                    val.clear()
                elif hasattr(val, '__del__'):
                    # Clear any objects with destructors (e.g., numpy arrays)
                    try:
                        del self._renderer._drillhole_polylines_cache[key]
                    except Exception:
                        pass
            self._renderer._drillhole_polylines_cache.clear()
            self._renderer._drillhole_polylines_cache = None
            
            # Force garbage collection for large datasets
            import gc
            gc.collect()
            logger.debug("[DRILLHOLE DEBUG] Cache cleared with memory released")
        
        try:
            self._renderer._update_scene_bounds()
        except Exception as exc:
            logger.debug(f"Could not refresh scene bounds after removing drillholes: {exc}")
        try:
            self._renderer._maintain_clipping_range()
        except Exception as e:
            logger.debug(f"Could not maintain clipping range: {e}")


    def get_drillhole_legend_metadata(self) -> Optional[Dict[str, Any]]:
        """Return the last computed drillhole legend metadata."""
        metadata = self._renderer._drillhole_legend_metadata
        if metadata:
            cats = metadata.get("categories", [])
            logger.info(f"[GET LEGEND METADATA] Returning metadata with {len(cats) if cats else 0} categories: {cats[:5] if cats else 'None'}")
        return metadata
    

    def set_drillhole_visibility(self, hole_id: str, visible: bool) -> None:
        """
        Toggle visibility for a single drillhole.
        
        Phase 3.2 Fix: Works in both merged and individual mode.
        In merged mode, rebuilds the mesh excluding hidden holes.
        
        Args:
            hole_id: The hole ID to toggle
            visible: Whether the hole should be visible
        """
        # Update state manager if available
        try:
            from ....core.state_manager import get_state_manager
            state_manager = get_state_manager()
            state_manager.set_drillhole_visibility(hole_id, visible)
        except Exception:
            pass
        
        # Check if we're in merged mode
        if "_merged" in self._renderer._drillhole_hole_actors:
            # Merged mode: need to rebuild with visibility filter
            self._rebuild_merged_drillholes_with_visibility()
        else:
            # Individual mode: toggle VTK actor visibility directly
            if hole_id in self._renderer._drillhole_hole_actors:
                actor = self._renderer._drillhole_hole_actors[hole_id]
                try:
                    if visible:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                except Exception as e:
                    logger.debug(f"Could not toggle drillhole visibility: {e}")
            
            # Also handle collar actor
            if hole_id in self._renderer._drillhole_collar_actors:
                collar_actor = self._renderer._drillhole_collar_actors[hole_id]
                try:
                    if visible:
                        collar_actor.VisibilityOn()
                    else:
                        collar_actor.VisibilityOff()
                except Exception as e:
                    logger.debug(f"Could not toggle collar visibility: {e}")
        
        # Render the changes
        if self._renderer.plotter is not None:
            try:
                self._renderer.plotter.render()
            except Exception:
                pass
    

    def _rebuild_merged_drillholes_with_visibility(self) -> None:
        """
        Rebuild merged drillhole mesh excluding hidden holes.
        
        Phase 3.2 Fix: When drillholes are merged for performance,
        visibility toggling requires rebuilding the merged geometry.
        """
        # Get visible holes from state manager
        try:
            from ....core.state_manager import get_state_manager
            state_manager = get_state_manager()
            visible_holes = state_manager.get_visible_holes()
        except Exception:
            # Fallback: get from layer data
            if "drillholes" in self._renderer.active_layers:
                layer_data = self._renderer.active_layers["drillholes"].get("data", {})
                visible_holes = set(layer_data.get("visible_holes", set()))
            else:
                return
        
        if not visible_holes:
            # Hide the merged actor entirely
            merged_actor = self._renderer._drillhole_hole_actors.get("_merged")
            if merged_actor:
                try:
                    merged_actor.VisibilityOff()
                except Exception:
                    pass
            return
        
        # Get cached data for rebuild
        cache = self._renderer._drillhole_polylines_cache
        if cache is None:
            logger.warning("No drillhole cache available for visibility rebuild")
            return
        
        hole_polys = cache.get("hole_polys", {})
        hole_segment_lith = cache.get("hole_segment_lith", {})
        lith_colors = cache.get("lith_colors", {})
        lith_to_index = cache.get("lith_to_index", {})
        
        if not hole_polys:
            return
        
        # Build meshes only for visible holes
        import pyvista as pv
        meshes_to_merge = []
        
        radius = getattr(self._renderer, '_drillhole_original_radius', 1.0)
        tube_resolution = 12  # Standard resolution
        
        for hid in visible_holes:
            if hid not in hole_polys:
                continue
            
            try:
                line = hole_polys[hid]
                if line is None or line.n_points < 2:
                    continue
                
                # Create tube
                tube = line.tube(radius=radius, n_sides=tube_resolution)
                
                # Apply coloring
                if hid in hole_segment_lith:
                    seg_liths = hole_segment_lith[hid]
                    lith_ids = np.array([lith_to_index.get(l, 0) for l in seg_liths], dtype=np.int32)
                    # Expand to tube cells
                    n_original_cells = len(seg_liths)
                    expansion = max(1, tube.n_cells // max(1, n_original_cells))
                    expanded = np.repeat(lith_ids, expansion)[:tube.n_cells]
                    if len(expanded) < tube.n_cells:
                        expanded = np.pad(expanded, (0, tube.n_cells - len(expanded)), constant_values=expanded[-1])
                    tube.cell_data['lith_id'] = expanded
                
                meshes_to_merge.append(tube)
            except Exception as e:
                logger.debug(f"Could not build tube for hole {hid}: {e}")
        
        if not meshes_to_merge:
            return
        
        # Merge all visible holes
        try:
            merged_mesh = pv.merge(meshes_to_merge)
        except Exception as e:
            logger.warning(f"Could not merge drillhole meshes: {e}")
            return
        
        # Remove old merged actor
        old_actor = self._renderer._drillhole_hole_actors.get("_merged")
        if old_actor:
            try:
                self._renderer.plotter.remove_actor(old_actor)
            except Exception:
                pass
        
        # Add new merged mesh
        try:
            scalar_name = "lith_id" if "lith_id" in merged_mesh.cell_data else None
            new_actor = self._renderer.plotter.add_mesh(
                merged_mesh,
                scalars=scalar_name,
                cmap="tab20",
                show_scalar_bar=False,
                name="drillholes_merged_visible"
            )
            self._renderer._drillhole_hole_actors["_merged"] = new_actor
            logger.debug(f"Rebuilt merged drillholes with {len(visible_holes)} visible holes")
        except Exception as e:
            logger.warning(f"Could not add merged drillhole mesh: {e}")
    

    def set_drillhole_labels_visible(self, visible: bool) -> None:
        """Show or hide drillhole ID labels."""
        if "drillholes" not in self._renderer.active_layers:
            return
        
        layer_data = self._renderer.active_layers["drillholes"].get("data", {})
        collar_coords = layer_data.get("collar_coords", {})
        
        if not collar_coords:
            return
        
        # Remove existing labels
        if hasattr(self._renderer, "_drillhole_label_actor") and self._renderer._drillhole_label_actor is not None:
            try:
                if self._renderer.plotter is not None:
                    self._renderer.plotter.remove_actor(self._renderer._drillhole_label_actor)
            except Exception:
                pass
            self._renderer._drillhole_label_actor = None
        
        if not visible or self._renderer.plotter is None:
            return
        
        # Add labels
        try:
            import numpy as np
            points = np.array(list(collar_coords.values()), dtype=float)
            labels = list(collar_coords.keys())
            
            if len(points) > 0:
                self._renderer._drillhole_label_actor = self._renderer.plotter.add_point_labels(
                    points,
                    labels,
                    font_size=10,
                    show_points=False,
                    shape=None,
                    text_color="black",
                    name="drillhole_labels",
                )
                if self._renderer.plotter is not None:
                    self._renderer.plotter.render()
        except Exception as e:
            logger.warning(f"Failed to add drillhole labels: {e}")
    

    def set_drillhole_visibility(self, hole_id: str, visible: bool) -> None:
        """Toggle visibility of a single drillhole and its collar marker (instant, no re-render)."""
        updated = False
        
        # ✅ FIX: Handle merged actor case (single actor for all holes)
        if "_merged" in self._renderer._drillhole_hole_actors:
            # Merged mode: can't toggle individual holes, but we track visibility state
            # and show/hide the merged actor based on whether ANY holes are visible
            # For now, just ensure merged actor is visible if any hole should be visible
            merged_actor = self._renderer._drillhole_hole_actors["_merged"]
            if visible:
                merged_actor.VisibilityOn()
                updated = True
            # Note: In merged mode, we don't hide if one hole is invisible
            # The merged actor stays visible as long as any hole should be visible
        elif hole_id in self._renderer._drillhole_hole_actors:
            actor = self._renderer._drillhole_hole_actors[hole_id]
            if visible:
                actor.VisibilityOn()
            else:
                actor.VisibilityOff()
            updated = True
        
        # Also toggle collar marker
        # In merged mode, collar is a single mesh - can't toggle individual collars
        if "_merged" in self._renderer._drillhole_collar_actors:
            # Merged collar actor stays visible as long as any holes are visible
            # Individual toggle not supported in merged mode
            pass
        elif hole_id in self._renderer._drillhole_collar_actors:
            collar_actor = self._renderer._drillhole_collar_actors[hole_id]
            if visible:
                collar_actor.VisibilityOn()
            else:
                collar_actor.VisibilityOff()
            updated = True
        
        # Update state manager
        if self._renderer._drillhole_state_manager:
            try:
                self._renderer._drillhole_state_manager.set_visibility(hole_id, visible)
            except Exception as e:
                logger.debug(f"State manager update failed: {e}")
        
        if updated and self._renderer.plotter is not None:
            self._renderer.plotter.render()
            logger.debug(f"Toggled drillhole {hole_id} visibility: {visible}")
        elif not updated:
            logger.warning(f"Drillhole {hole_id} not found in actors cache")
    

    def set_all_drillholes_visible(self, visible_holes: Set[str]) -> None:
        """
        ULTRA-FAST batch visibility update.
        
        Uses VTK visibility flags (no geometry rebuild).
        Target: <5ms for 200 holes.
        """
        import time
        start = time.perf_counter()
        
        if self._renderer.plotter is None:
            return
        
        updated = 0
        
        # ✅ FIX: Handle merged actor case (single actor for all holes)
        if "_merged" in self._renderer._drillhole_hole_actors:
            merged_actor = self._renderer._drillhole_hole_actors["_merged"]
            # Show merged actor if ANY holes should be visible
            should_show = len(visible_holes) > 0
            current_visible = merged_actor.GetVisibility() > 0
            if should_show != current_visible:
                if should_show:
                    merged_actor.VisibilityOn()
                else:
                    merged_actor.VisibilityOff()
                updated += 1
                logger.debug(f"Merged drillhole actor visibility: {should_show}")
        else:
            # Individual actors mode
            for hole_id, actor in self._renderer._drillhole_hole_actors.items():
                should_show = hole_id in visible_holes
                current_visible = actor.GetVisibility() > 0
                if should_show != current_visible:
                    if should_show:
                        actor.VisibilityOn()
                    else:
                        actor.VisibilityOff()
                    updated += 1
        
        # Update collar markers
        if "_merged" in self._renderer._drillhole_collar_actors:
            # Merged collar mode - show/hide based on whether ANY holes are visible
            merged_collar = self._renderer._drillhole_collar_actors["_merged"]
            should_show = len(visible_holes) > 0
            current_visible = merged_collar.GetVisibility() > 0
            if should_show != current_visible:
                if should_show:
                    merged_collar.VisibilityOn()
                else:
                    merged_collar.VisibilityOff()
        else:
            # Individual collar actors mode
            for hole_id, collar_actor in self._renderer._drillhole_collar_actors.items():
                if hole_id.startswith("_"):
                    continue  # Skip metadata keys
                should_show = hole_id in visible_holes
                current_visible = collar_actor.GetVisibility() > 0
                if should_show != current_visible:
                    if should_show:
                        collar_actor.VisibilityOn()
                    else:
                        collar_actor.VisibilityOff()
        
        # Update state manager
        if self._renderer._drillhole_state_manager:
            try:
                # Get hole IDs from merged metadata or collar keys
                if "_merged_hids" in self._renderer._drillhole_collar_actors:
                    all_hole_ids = set(self._renderer._drillhole_collar_actors["_merged_hids"])
                else:
                    all_hole_ids = set(k for k in self._renderer._drillhole_collar_actors.keys() if not k.startswith("_"))
                for hole_id in all_hole_ids:
                    self._renderer._drillhole_state_manager.set_visibility(hole_id, hole_id in visible_holes)
            except Exception as e:
                logger.debug(f"State manager batch update failed: {e}")
        
        # Single render for all updates
        if updated > 0 and self._renderer.plotter is not None:
            self._renderer.plotter.render()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"FAST visibility update: {updated} holes in {elapsed:.1f}ms")
    

    def update_drillhole_selection(self, selected_hole_ids: Set[str]) -> None:
        """
        Update drillhole selection state for both rendering and visual density control.

        This method should be called whenever drillhole selections change in the UI.

        Args:
            selected_hole_ids: Set of currently selected hole IDs
        """
        # Update visual density controller (selected holes stay full detail)
        self._renderer.update_selected_drillholes_for_density(selected_hole_ids)

        # If using GPU renderer, update its selection state too
        if hasattr(self._renderer, '_gpu_drillhole_renderer') and self._renderer._gpu_drillhole_renderer is not None:
            try:
                # Convert hole IDs to color IDs for GPU renderer
                selected_color_ids = set()
                for hole_id in selected_hole_ids:
                    intervals = self._renderer._gpu_drillhole_renderer.state.get_intervals_by_hole(hole_id)
                    selected_color_ids.update(iv.color_id for iv in intervals)

                if selected_color_ids:
                    self._renderer._gpu_drillhole_renderer.update_selection(selected_color_ids)
            except Exception as e:
                logger.debug(f"Failed to update GPU renderer selection: {e}")

        logger.debug(f"Updated drillhole selection: {len(selected_hole_ids)} holes selected")


    def update_drillhole_radius(self, radius: float) -> None:
        """
        ACTOR PERSISTENCE: Update geometry in-place without removing actors.
        
        CRITICAL FIX: To keep picking stable, we MUST NOT remove/re-add actors.
        Instead, we update the VBO (Vertex Buffer Object) directly by copying
        new geometry into the existing actor's mapper.
        
        This prevents:
        - Picking drift (actor IDs stay constant)
        - Video jitter (no actor removal/creation mid-frame)
        - Selection loss (actor references remain valid)
        """
        if self._renderer._drillhole_polylines_cache is None:
            return
        
        if self._renderer.plotter is None:
            logger.warning("Cannot update radius: plotter not initialized")
            return
        
        cache = self._renderer._drillhole_polylines_cache
        
        # Loop protection: Check if radius is effectively unchanged
        current_radius = cache.get("radius", -1.0)
        if abs(current_radius - radius) < 1e-4:
            logger.debug(f"Drillhole radius {radius} unchanged, skipping update")
            return
        
        # Update cache radius
        cache["radius"] = radius
        
        # Check if we have a merged actor (most common case)
        if "_merged" not in self._renderer._drillhole_hole_actors:
            logger.debug("No merged drillhole actor found, falling back to full rebuild")
            self.update_drillhole_radius_full_rebuild(radius)
            return
        
        actor = self._renderer._drillhole_hole_actors["_merged"]
        hole_polys = cache["hole_polys"]
        
        # Extract color data from cache for scalar mapping
        hole_segment_lith = cache.get("hole_segment_lith", {})
        hole_segment_assay = cache.get("hole_segment_assay", {})
        lith_to_index = cache.get("lith_to_index", {})
        color_mode = cache.get("color_mode", "Lithology")
        scalar_name = "lith_id" if color_mode == "Lithology" else "assay"
        
        # Rebuild spline tubes with new radius, preserving cell scalars
        n_sides = 16
        total = len(hole_polys)
        if total > 500:
            n_sides = 12
        elif total > 200:
            n_sides = 14

        hide_barren = getattr(self._renderer, '_hide_barren_intervals', True)

        tubes_to_merge = []
        for hid, poly in hole_polys.items():
            if poly is None or poly.n_cells < 1:
                continue
            # Prepare per-segment scalars
            if color_mode == "Lithology":
                seg_scalars = [float(lith_to_index.get(lit, -1))
                               for lit in hole_segment_lith.get(hid, [])]
            else:
                raw_scalars = list(hole_segment_assay.get(hid, []))
                seg_scalars = []
                for v in raw_scalars:
                    if v is not None and np.isfinite(v):
                        if hide_barren and v <= 0:
                            seg_scalars.append(float('nan'))
                        else:
                            seg_scalars.append(v)
                    else:
                        seg_scalars.append(float('nan'))

            # Skip entirely-barren holes when hiding barren
            if hide_barren and seg_scalars and all(
                (np.isnan(v) if isinstance(v, float) else False) for v in seg_scalars
            ):
                continue

            tube = self._build_spline_tube(poly, radius, n_sides, seg_scalars, scalar_name)
            if tube is not None and tube.n_cells > 0:
                tubes_to_merge.append(tube)

        if not tubes_to_merge:
            logger.warning("No polylines to update")
            return

        try:
            if len(tubes_to_merge) == 1:
                new_mesh = tubes_to_merge[0]
            else:
                new_mesh = pv.merge(tubes_to_merge)
            
            # NOTE: Do NOT apply _to_local_precision here!
            # The cached polylines are in world coordinates, same as the initial render.
            # Applying _to_local_precision would double-shift the geometry, causing
            # drillholes to disappear when radius is updated.
            
            # IN-PLACE UPDATE: Set new geometry on the existing actor's mapper
            # This keeps the actor identity stable for picking
            mapper = actor.GetMapper()
            
            # Preserve the existing lookup table before updating input data
            existing_lut = mapper.GetLookupTable()
            
            mapper.SetInputData(new_mesh)
            
            # CRITICAL: Configure mapper to use the scalar array for coloring
            # Without this, the mapper loses color configuration after SetInputData
            if scalar_name in new_mesh.point_data:
                mapper.SetScalarModeToUsePointData()
                mapper.SelectColorArray(scalar_name)
                mapper.ScalarVisibilityOn()
                
                # Restore the lookup table (colormap) that was configured initially
                if existing_lut is not None:
                    mapper.SetLookupTable(existing_lut)
                    # Update NaN color alpha based on hide_barren
                    if hide_barren:
                        existing_lut.SetNanColor(0.5, 0.5, 0.5, 0.0)
                    else:
                        existing_lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
                    existing_lut.Build()
                    # Update scalar range for continuous data (assay mode)
                    if color_mode != "Lithology":
                        assay_min = cache.get("assay_min", 0.0)
                        assay_max = cache.get("assay_max", 1.0)
                        mapper.SetScalarRange(assay_min, assay_max)
            
            mapper.Modified()
            actor.Modified()
            
            logger.info(f"Updated drillhole radius to {radius} (in-place, actor preserved, colors retained)")
            
            # Force render
            if self._renderer.plotter:
                self._renderer.plotter.render()
                
        except Exception as e:
            logger.warning(f"In-place update failed: {e}, falling back to full rebuild")
            self.update_drillhole_radius_full_rebuild(radius)
    

    def update_drillhole_radius_full_rebuild(self, radius: float) -> None:
        """
        FALLBACK: Full geometry rebuild for radius update (slow but accurate).
        
        Use this if transform-based scaling causes issues.
        Updates cache and layer data to keep radius synchronized.
        """
        import time
        start = time.perf_counter()
        
        if self._renderer._drillhole_polylines_cache is None:
            logger.warning("Cannot update radius: no cached polylines")
            return
        
        if self._renderer.plotter is None:
            logger.warning("Cannot update radius: plotter not initialized")
            return
        
        cache = self._renderer._drillhole_polylines_cache
        
        # Loop protection: Check if radius is effectively unchanged
        current_radius = cache.get("radius", -1.0)
        if abs(current_radius - radius) < 1e-4:
            logger.debug(f"Drillhole radius {radius} unchanged, skipping rebuild")
            return
            
        # CRITICAL: Update cache radius immediately to keep in sync
        cache["radius"] = radius
        
        # Also update layer data
        if "drillholes" in self._renderer.active_layers:
            self._renderer.active_layers["drillholes"]["data"]["radius"] = radius
        
        hole_polys = cache["hole_polys"]
        hole_segment_lith = cache["hole_segment_lith"]
        hole_segment_assay = cache["hole_segment_assay"]
        lith_colors = cache["lith_colors"]
        lith_to_index = cache["lith_to_index"]
        assay_min = cache["assay_min"]
        assay_max = cache["assay_max"]
        color_mode = cache["color_mode"]
        collar_coords = cache["collar_coords"]
        
        # Get current colormap from legend metadata or use default
        current_colormap = "viridis"
        if self._renderer._drillhole_legend_metadata:
            current_colormap = self._renderer._drillhole_legend_metadata.get("colormap", "viridis")
        
        scalar_name = "lith_id" if color_mode == "Lithology" else "assay"
        
        # Remove old actors
        for hole_id, actor in self._renderer._drillhole_hole_actors.items():
            try:
                self._renderer.plotter.remove_actor(actor)
            except Exception:
                pass
        
        self._renderer._drillhole_hole_actors.clear()
        
        # Get current visible holes from layer data
        visible_holes = set()
        if "drillholes" in self._renderer.active_layers:
            layer_data = self._renderer.active_layers["drillholes"].get("data", {})
            visible_holes = set(layer_data.get("visible_holes", set()))
        
        # Adaptive quality based on dataset size
        # Minimum 12 sides ensures tubes look cylindrical, not polygonal
        total_holes = len(hole_polys)
        n_sides = 16
        if total_holes > 500:
            n_sides = 12
        elif total_holes > 200:
            n_sides = 14
        
        # Build spline-smoothed tubes per hole with cell scalars
        hide_barren = getattr(self._renderer, '_hide_barren_intervals', True)
        tubes_to_merge = []

        for hid in hole_polys.keys():
            poly = hole_polys.get(hid)
            if poly is None or poly.n_cells < 1:
                continue
            if color_mode == "Lithology":
                seg_scalars = [float(lith_to_index.get(lit, -1))
                               for lit in hole_segment_lith.get(hid, [])]
            else:
                raw_scalars = list(hole_segment_assay.get(hid, []))
                seg_scalars = []
                for v in raw_scalars:
                    if v is not None and np.isfinite(v):
                        if hide_barren and v <= 0:
                            seg_scalars.append(float('nan'))
                        else:
                            seg_scalars.append(v)
                    else:
                        seg_scalars.append(float('nan'))

            # Skip entirely-barren holes when hiding barren
            if hide_barren and seg_scalars and all(
                (np.isnan(v) if isinstance(v, float) else False) for v in seg_scalars
            ):
                continue

            tube = self._build_spline_tube(poly, radius, n_sides, seg_scalars, scalar_name)
            if tube is not None and tube.n_cells > 0:
                tubes_to_merge.append(tube)

        if not tubes_to_merge:
            logger.warning("No polylines to render after filtering")
            return

        assay_p98 = cache.get("assay_p98", assay_max)
        nan_clr = (0.5, 0.5, 0.5, 0.0) if hide_barren else "gray"

        try:
            merged = tubes_to_merge[0] if len(tubes_to_merge) == 1 else pv.merge(tubes_to_merge)
            logger.info(f"Radius update: merged {len(tubes_to_merge)} tubes ({merged.n_cells} cells)")
        except Exception as e:
            logger.error(f"Tube merge failed: {e}")
            return

        if color_mode == "Lithology":
            lith_cmap = self._renderer._build_lithology_colormap(current_colormap, lith_colors, lith_to_index)
            actor = self._renderer.plotter.add_mesh(
                merged, scalars=scalar_name, cmap=lith_cmap,
                show_scalar_bar=False, reset_camera=False,
                smooth_shading=True, pbr=True, metallic=0.1, roughness=0.5,
                nan_color=nan_clr, name="drillholes_batched",
            )
        else:
            actor = self._renderer.plotter.add_mesh(
                merged, scalars=scalar_name, clim=[assay_min, assay_p98],
                cmap=current_colormap, show_scalar_bar=False, reset_camera=False,
                smooth_shading=True, pbr=True, metallic=0.1, roughness=0.5,
                nan_color=nan_clr, name="drillholes_batched",
            )

        self._renderer._drillhole_merged_actor = actor
        self._renderer._drillhole_merged_mesh = merged
        self._renderer._drillhole_hole_actors = {"_merged": actor}
        actor.VisibilityOn()
        actor.Modified()

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Radius update completed: {len(tubes_to_merge)} holes in {elapsed:.1f}ms")

        # Rebuild collar spheres with new radius
        if hasattr(self._renderer, '_drillhole_collar_actors'):
            for key, collar_actor in list(self._renderer._drillhole_collar_actors.items()):
                if key == "_merged_hids":
                    continue
                try:
                    self._renderer.plotter.remove_actor(collar_actor)
                except Exception:
                    pass
            self._renderer._drillhole_collar_actors.clear()

            collar_radius = radius * 0.6
            collar_pts = []
            visible_collar_hids = []
            for hid in hole_polys.keys():
                collar = collar_coords.get(hid)
                if collar is None:
                    continue
                if visible_holes and hid not in visible_holes:
                    continue
                visible_collar_hids.append(hid)
                collar_pts.append(collar)

            if collar_pts:
                collar_cloud = pv.PolyData(np.array(collar_pts, dtype=np.float64))
                collar_glyphs = collar_cloud.glyph(
                    geom=pv.Sphere(radius=collar_radius), orient=False, scale=False,
                )
                collar_actor = self._renderer.plotter.add_mesh(
                    collar_glyphs, color="white", smooth_shading=True,
                    pbr=True, metallic=0.3, roughness=0.4,
                    show_scalar_bar=False, reset_camera=False,
                    name="drillhole_collars_merged",
                )
                collar_actor.VisibilityOn()
                self._renderer._drillhole_collar_actors["_merged"] = collar_actor
                self._renderer._drillhole_collar_actors["_merged_hids"] = visible_collar_hids
        
        # Update layer data
        if "drillholes" in self._renderer.active_layers:
            self._renderer.active_layers["drillholes"]["data"]["radius"] = radius
        
        logger.info(f"Updated drillhole radius to {radius} for {len(self._renderer._drillhole_hole_actors)} holes")

        if self._renderer.plotter is not None:
            self._renderer.plotter.render()

        # Re-apply SSAO/EDL — add_mesh resets VTK render passes
        self._renderer.reapply_scene_effects()

