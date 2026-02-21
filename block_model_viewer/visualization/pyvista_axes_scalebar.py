"""
3D Axes Box and Dynamic Scale Bar with PyVista.

Provides FloatingAxes and ScaleBar3D classes for professional 3D visualization
with tick marks, labels, and dynamic scale bars.
"""

import pyvista as pv
import numpy as np
import logging
from typing import Optional, Tuple, List, Callable
from matplotlib import colors as mcolors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkLine
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper2D, vtkActor2D, vtkTextActor

logger = logging.getLogger(__name__)


class FloatingAxes:
    """
    Floating 3D axes box with tick marks and labels.
    
    Creates a 3D axes box using PyVista primitives with major and minor tick marks
    and 3D text labels for tick values.
    """
    
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float],
                 x_major: float = 10, x_minor: float = 5,
                 y_major: float = 10, y_minor: float = 5,
                 z_major: float = 10, z_minor: float = 5,
                 major_tick_length: float = 2.0, minor_tick_length: float = 1.0,
                 axis_line_width: float = 2.0, tick_line_width: float = 1.0,
                 axis_color: str = 'white', tick_color: str = 'white', label_color: str = 'white',
                 label_size: float = 5.0,
                 label_offset: Optional[np.ndarray] = None):
        """
        Initialize a floating 3D axes box with tick marks and labels.

        Args:
            x_range, y_range, z_range: (min, max) for each axis (local/geometry coordinates).
            x_major, x_minor, ...: spacing for major/minor ticks on each axis.
            major_tick_length, minor_tick_length: lengths of tick marks (world units).
            axis_line_width, tick_line_width: line widths for axes and ticks.
            axis_color, tick_color, label_color: colors for axes, ticks, labels.
            label_size: height of tick label text (world units).
            label_offset: Optional [x, y, z] offset to add to tick labels for displaying
                         world/UTM coordinates instead of local coordinates.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        # Label offset for displaying world coordinates (added to local tick values)
        self.label_offset = label_offset if label_offset is not None else np.zeros(3)
        self.spacings = {
            'x': {'major': x_major, 'minor': x_minor},
            'y': {'major': y_major, 'minor': y_minor},
            'z': {'major': z_major, 'minor': z_minor}
        }
        self.major_tick_length = major_tick_length
        self.minor_tick_length = minor_tick_length
        self.axis_line_width = axis_line_width
        self.tick_line_width = tick_line_width
        self.axis_color = axis_color
        self.tick_color = tick_color
        self.label_color = label_color
        self.label_size = label_size
        
        # Compute origin (min corner) and axis end points
        self.origin = np.array([x_range[0], y_range[0], z_range[0]], dtype=float)
        self.ends = {
            'x': np.array([x_range[1], y_range[0], z_range[0]], dtype=float),
            'y': np.array([x_range[0], y_range[1], z_range[0]], dtype=float),
            'z': np.array([x_range[0], y_range[0], z_range[1]], dtype=float)
        }
        
        # Storage for created meshes
        self.axis_lines_mesh: Optional[pv.PolyData] = None
        self.major_ticks_mesh: Optional[pv.PolyData] = None
        self.minor_ticks_mesh: Optional[pv.PolyData] = None
        self.labels_mesh: Optional[pv.PolyData] = None
        self._actors: List = []

    def _format_axis_label(self, value: float) -> str:
        """
        Format axis label value using compact notation for large numbers.

        Large UTM coordinates (e.g., 500000) are formatted as "500k" to fit
        in the available space. Small values are shown with appropriate precision.
        """
        abs_val = abs(value)
        if abs_val >= 1_000_000:
            # Millions: show as "1.5M"
            return f"{value / 1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif abs_val >= 10_000:
            # Ten-thousands: show as "500k"
            return f"{value / 1_000:.0f}k"
        elif abs_val >= 1_000:
            # Thousands: show as "1.5k"
            return f"{value / 1_000:.1f}k".rstrip('0').rstrip('.')
        elif abs_val >= 100:
            return f"{value:.0f}"
        elif abs_val >= 10:
            return f"{value:.0f}"
        elif abs_val >= 1:
            return f"{value:.1f}".rstrip('0').rstrip('.')
        elif abs_val > 0:
            return f"{value:.2f}".rstrip('0').rstrip('.')
        else:
            return "0"

    def _generate_ticks_for_axis(self, axis: str) -> Tuple[List[Tuple], List[Tuple]]:
        """Compute positions of major and minor tick segments for a given axis ('x','y','z')."""
        if axis == 'x':
            axis_start = np.array([self.x_range[0], self.y_range[0], self.z_range[0]], dtype=float)
            axis_end = np.array([self.x_range[1], self.y_range[0], self.z_range[0]], dtype=float)
            tick_dir = np.array([0, -1, 0], dtype=float)  # ticks point in -Y (front-facing)
        elif axis == 'y':
            axis_start = np.array([self.x_range[0], self.y_range[0], self.z_range[0]], dtype=float)
            axis_end = np.array([self.x_range[0], self.y_range[1], self.z_range[0]], dtype=float)
            tick_dir = np.array([-1, 0, 0], dtype=float)  # ticks point in -X (left-facing)
        elif axis == 'z':
            axis_start = np.array([self.x_range[0], self.y_range[0], self.z_range[0]], dtype=float)
            axis_end = np.array([self.x_range[0], self.y_range[0], self.z_range[1]], dtype=float)
            tick_dir = np.array([0, 0, -1], dtype=float)  # ticks point downward (outside bottom)
        else:
            return [], []
        
        # Axis vector and length
        axis_vec = axis_end - axis_start
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1e-6:
            return [], []
        
        # Determine tick positions along the axis
        major_step = self.spacings[axis]['major'] or axis_len
        minor_step = self.spacings[axis]['minor'] or 0
        
        min_val = 0.0
        max_val = axis_len
        
        # Align first major tick at or beyond start
        start = np.ceil(min_val / major_step) * major_step
        if abs(start - min_val) < 1e-6:
            start = min_val  # include origin if exactly on a tick
        
        major_positions = list(np.arange(start, max_val + 1e-6, major_step))
        if abs(major_positions[-1] - max_val) > 1e-6:
            major_positions.append(max_val)  # ensure tick at axis end
        
        minor_positions = []
        if minor_step > 0:
            minor_positions = list(np.arange(np.ceil(min_val / minor_step) * minor_step, max_val + 1e-6, minor_step))
            # Exclude positions coincident with major ticks
            minor_positions = [m for m in minor_positions if all(abs(m - M) > 1e-6 for M in major_positions)]
        
        # Create line segments for each tick, including tick value for labeling
        major_segments = []
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        axis_offset = axis_start[axis_index]
        # Get the label offset for this axis (converts local to world coordinates)
        world_offset = self.label_offset[axis_index] if self.label_offset is not None else 0.0

        for t in major_positions:
            point = axis_start + (axis_vec / axis_len) * t
            # Add world offset to display UTM/world coordinates instead of local
            label_value = axis_offset + t + world_offset
            major_segments.append((point, point + tick_dir * self.major_tick_length, label_value))

        minor_segments = []
        for t in minor_positions:
            point = axis_start + (axis_vec / axis_len) * t
            # Add world offset to display UTM/world coordinates instead of local
            label_value = axis_offset + t + world_offset
            minor_segments.append((point, point + tick_dir * self.minor_tick_length, label_value))
        
        return major_segments, minor_segments
    
    def add_to_plotter(self, plotter: pv.Plotter, draw_box: bool = True) -> None:
        """Add the axes lines, ticks, and labels to a PyVista Plotter."""
        # Clear any existing actors
        self.clear(plotter)
        
        # 1. Create axes lines (either 3 axes or full box frame)
        segments = [
            (self.origin, self.ends['x']),
            (self.origin, self.ends['y']),
            (self.origin, self.ends['z'])
        ]
        
        if draw_box:
            xmin, xmax = self.x_range
            ymin, ymax = self.y_range
            zmin, zmax = self.z_range
            # Add remaining 9 edges of the box (total 12 edges including axes)
            segments += [
                ([xmin, ymin, zmin], [xmax, ymin, zmin]),  # bottom face X-edges
                ([xmin, ymax, zmin], [xmax, ymax, zmin]),
                ([xmin, ymin, zmin], [xmin, ymax, zmin]),  # bottom face Y-edges
                ([xmax, ymin, zmin], [xmax, ymax, zmin]),
                ([xmin, ymin, zmax], [xmax, ymin, zmax]),  # top face X-edges
                ([xmin, ymax, zmax], [xmax, ymax, zmax]),
                ([xmin, ymin, zmax], [xmin, ymax, zmax]),  # top face Y-edges
                ([xmax, ymin, zmax], [xmax, ymax, zmax]),
                ([xmin, ymin, zmin], [xmin, ymin, zmax]),  # vertical edges
                ([xmax, ymin, zmin], [xmax, ymin, zmax]),
                ([xmin, ymax, zmin], [xmin, ymax, zmax]),
                ([xmax, ymax, zmin], [xmax, ymax, zmax])
            ]
        
        # Merge all axis segments into one mesh for performance
        combined_axes = None
        for p1, p2 in segments:
            line = pv.Line(p1, p2)
            combined_axes = line if combined_axes is None else combined_axes.merge(line)
        
        self.axis_lines_mesh = combined_axes
        if combined_axes is not None:
            actor = plotter.add_mesh(
                self.axis_lines_mesh,
                color=self.axis_color,
                line_width=self.axis_line_width,
                pickable=False
            )
            self._actors.append(actor)
        
        # 2. Generate and add tick mark segments
        all_major_ticks = None
        all_minor_ticks = None
        label_mesh = None
        
        for axis in ('x', 'y', 'z'):
            major_segs, minor_segs = self._generate_ticks_for_axis(axis)
            
            # Merge major tick lines
            for seg in major_segs:
                if len(seg) == 3:
                    p1, p2, tick_value = seg
                else:
                    # Backward compatibility
                    p1, p2 = seg[:2]
                    tick_value = 0.0
                
                line = pv.Line(p1, p2)
                all_major_ticks = line if all_major_ticks is None else all_major_ticks.merge(line)
                
                # Create 3D text label at the end of the major tick
                # Format label - use compact notation for large numbers
                label_str = self._format_axis_label(tick_value)

                # Position label slightly beyond tick end for clarity
                tick_dir_vec = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)
                label_pos = p2 + tick_dir_vec * (0.8 * self.major_tick_length)

                # Create text oriented outward (normal = tick_dir)
                # Use smaller label size for readability
                effective_label_size = min(self.label_size, self.major_tick_length * 0.8)
                try:
                    text = pv.Text3D(
                        label_str,
                        depth=0.0,
                        height=effective_label_size,
                        center=label_pos,
                        normal=tick_dir_vec
                    )
                    label_mesh = text if label_mesh is None else label_mesh.merge(text)
                except Exception as e:
                    logger.debug(f"Failed to create text label for {axis} axis tick: {e}")
            
            # Merge minor tick lines
            for seg in minor_segs:
                if len(seg) == 3:
                    p1, p2, _ = seg
                else:
                    # Backward compatibility
                    p1, p2 = seg[:2]
                line = pv.Line(p1, p2)
                all_minor_ticks = line if all_minor_ticks is None else all_minor_ticks.merge(line)
        
        # Add tick mark meshes to the plotter
        if all_major_ticks:
            self.major_ticks_mesh = all_major_ticks
            actor = plotter.add_mesh(
                self.major_ticks_mesh,
                color=self.tick_color,
                line_width=self.tick_line_width,
                pickable=False
            )
            self._actors.append(actor)
        
        if all_minor_ticks:
            self.minor_ticks_mesh = all_minor_ticks
            actor = plotter.add_mesh(
                self.minor_ticks_mesh,
                color=self.tick_color,
                line_width=self.tick_line_width * 0.8,
                pickable=False
            )
            self._actors.append(actor)
        
        # Add merged label mesh
        if label_mesh:
            self.labels_mesh = label_mesh
            actor = plotter.add_mesh(self.labels_mesh, color=self.label_color, pickable=False)
            self._actors.append(actor)
    
    def clear(self, plotter: pv.Plotter) -> None:
        """Remove all axes actors from the plotter."""
        for actor in self._actors:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._actors.clear()
        self.axis_lines_mesh = None
        self.major_ticks_mesh = None
        self.minor_ticks_mesh = None
        self.labels_mesh = None

    def get_actors(self) -> List[pv.Actor]:
        """Return the current PyVista actors used by the floating axes."""
        return [actor for actor in self._actors if actor is not None]


class ScaleBar3D:
    """
    Dynamic 3D scale bar with graduated tick marks and labels.
    """
    
    def __init__(
        self,
        units: str = "m",
        bar_fraction: float = 0.2,
        color: str = 'white',
        text_color: str = 'white',
        line_width: float = 2.0,
        text_height: Optional[float] = None,
        position_x: float = 0.8,
        position_y: float = 0.05,
        reference_actor_fn: Optional[Callable[[pv.Plotter], List]] = None
    ):
        """
        Initialize a dynamic 3D scale bar with tick marks.
        
        Args:
            units: label for units (e.g., "m" for meters).
            bar_fraction: fraction of the view width that the scale bar should span.
            color: line color for the scale bar and ticks.
            text_color: color for the scale bar label text.
            line_width: thickness of the scale bar line.
            text_height: height of the label text (world units). If None, size will adjust with zoom.
            position_x / position_y: normalized screen anchor (0-1) used when no reference actor.
            reference_actor_fn: optional callback to supply actors (e.g., floating axes) the bar
                should sit underneath in screen space.
        """
        self.units = units
        self.bar_fraction = bar_fraction
        self.color = color
        self.text_color = text_color
        self.line_width = line_width
        self.text_height = text_height
        self.position_x = position_x
        self.position_y = position_y
        self._reference_actor_fn = reference_actor_fn
        self._user_anchor: Optional[Tuple[float, float]] = None
        self._dragging = False
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._last_display_origin: Optional[Tuple[float, float]] = None
        self._last_pixel_length: float = 0.0
        self._last_bar_bounds: Optional[Tuple[float, float, float, float]] = None
        self._interaction_installed = False
        self._interactor = None
        self._press_tag = None
        self._move_tag = None
        self._release_tag = None
        self._plotter_ref: Optional[pv.Plotter] = None
        
        # Actors for the scale bar components
        self.bar_actor: Optional[pv.Actor] = None
        self.major_tick_actors: List[pv.Actor] = []
        self.minor_tick_actors: List[pv.Actor] = []
        self.label_actors: List[pv.Actor] = []
        self.unit_label_actor: Optional[pv.Actor] = None
        self._callback_registered = False
        self._current_display_length: float = 0.0

    def _get_unit_conversion(self) -> float:
        """Get conversion factor from meters (source) to display units."""
        unit = self.units.lower().strip()
        conversions = {
            'm': 1.0,
            'meters': 1.0,
            'km': 0.001,
            'kilometers': 0.001,
            'ft': 3.28084,
            'feet': 3.28084,
            'mm': 1000.0,
            'millimeters': 1000.0,
            'miles': 0.000621371,
            'mi': 0.000621371,
        }
        return conversions.get(unit, 1.0)
        self._camera_observer_tag = None
        self._interactor_observers_registered = False
        self._update_pending = False

    def add_to_plotter(self, plotter: pv.Plotter) -> None:
        """Add the scale bar to the plotter and set up the render callback for dynamic updating."""
        logger.info(f"ScaleBar3D.add_to_plotter called, plotter={plotter}")

        # Store plotter reference FIRST so callbacks can access it
        self.attach_plotter(plotter)
        logger.info(f"Plotter attached, _plotter_ref={self._plotter_ref}")

        # Register the update callback on each render event
        # Use a wrapper since PyVista callbacks don't pass arguments
        if not self._callback_registered:
            def _update_callback(*args):
                if self._plotter_ref is not None:
                    self.update(self._plotter_ref)
            self._update_callback_fn = _update_callback  # Keep reference to prevent garbage collection
            plotter.add_on_render_callback(_update_callback, render_event=True)
            self._callback_registered = True
            logger.info("Render callback registered")

        # Note: VTK observers don't work with pyvistaqt's interactor
        # Zoom updates are handled via Qt wheel events in viewer_widget.py

        # Do an initial update to set correct size/position
        logger.info("Calling initial update...")
        self.update(plotter)
        logger.info("Initial update completed")
    
    def update(self, plotter: pv.Plotter) -> None:
        """Update the scale bar position and label based on the current camera view."""
        try:
            logger.debug(f"ScaleBar3D.update called")
            cam = plotter.camera  # active vtkCamera
            pos = np.array(cam.GetPosition())
            focal = np.array(cam.GetFocalPoint())
            logger.debug(f"Camera pos={pos}, focal={focal}")
            
            # Camera distance drives projected width/height; orientation no longer affects 2D overlay
            forward = focal - pos
            dist = np.linalg.norm(forward)
            if dist < 1e-6:
                return  # degenerate camera position
            
            # Determine view width (world units) at focal plane
            window_size = plotter.window_size
            if window_size[0] == 0 or window_size[1] == 0:
                return
            
            aspect = window_size[0] / window_size[1]
            
            if cam.GetParallelProjection():
                # Orthographic: use parallel scale (half height in world units)
                height = cam.GetParallelScale() * 2.0
            else:
                # Perspective: use camera view angle (vertical FOV) to get height at distance
                fov = cam.GetViewAngle()  # vertical field-of-view in degrees
                height = 2 * dist * np.tan(np.deg2rad(fov) / 2.0)
            
            width = height * aspect
            
            # Ideal scale bar length as a fraction of view width
            ideal = self.bar_fraction * width
            if ideal <= 0:
                return
            
            # Validate against scene bounds to ensure reasonable scale bar length
            # Get scene bounds from plotter actors (excluding overlay actors like scale bar and axes)
            try:
                renderer = plotter.renderer
                if renderer:
                    # Get all actors and compute bounds excluding overlay actors
                    actors = renderer.GetActors()
                    actors.InitTraversal()
                    actor = actors.GetNextItem()
                    
                    bounds_list = []
                    while actor:
                        # Skip overlay actors (scale bar, floating axes) by checking if they're pickable=False
                        # and have specific properties, or check actor name/tag
                        try:
                            # Get actor bounds
                            actor_bounds = actor.GetBounds()
                            if actor_bounds and len(actor_bounds) >= 6:
                                # Check if bounds are reasonable (not too large, likely overlay)
                                span = max(
                                    actor_bounds[1] - actor_bounds[0],
                                    actor_bounds[3] - actor_bounds[2],
                                    actor_bounds[5] - actor_bounds[4]
                                )
                                # Skip actors with extremely large bounds (likely overlays)
                                if span < 1_000_000:  # Reasonable threshold
                                    bounds_list.append(actor_bounds)
                        except Exception:
                            pass
                        actor = actors.GetNextItem()
                    
                    if bounds_list:
                        # Compute overall bounds from valid actors
                        all_bounds = np.array(bounds_list)
                        min_bounds = all_bounds[:, [0, 2, 4]].min(axis=0)
                        max_bounds = all_bounds[:, [1, 3, 5]].max(axis=0)
                        
                        scene_span_x = max_bounds[0] - min_bounds[0]
                        scene_span_y = max_bounds[1] - min_bounds[1]
                        scene_span_z = max_bounds[2] - min_bounds[2]
                        max_scene_span = max(scene_span_x, scene_span_y, scene_span_z)
                        
                        # Constrain ideal length to be at most 50% of scene span
                        # This prevents the scale bar from showing unrealistic values
                        if max_scene_span > 0 and max_scene_span < 1_000_000:
                            max_reasonable_length = max_scene_span * 0.5
                            ideal = min(ideal, max_reasonable_length)
            except Exception:
                # If we can't get bounds, continue with original calculation
                pass
            
            # Convert ideal length from meters to display units
            unit_factor = self._get_unit_conversion()
            ideal_display = ideal * unit_factor

            # Choose a "nice" round number in DISPLAY units (1, 2, or 5 times 10^n)
            if ideal_display > 0:
                exp = int(np.floor(np.log10(ideal_display)))
                base = 10 ** exp
                candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
                # Pick the closest candidate to ideal length in display units
                display_length_val = candidates[np.abs(candidates - ideal_display).argmin()]
            else:
                display_length_val = ideal_display

            if display_length_val <= 0:
                display_length_val = ideal_display  # fallback if something went wrong

            # Convert back to meters for pixel calculations
            length_val_meters = display_length_val / unit_factor if unit_factor > 0 else display_length_val

            # Store display value for labels
            self._current_display_length = display_length_val

            # Compute pixel length of scale bar (using meters for accurate sizing)
            display_width = window_size[0]
            pixel_length = max(30.0, length_val_meters / width * display_width)
            anchor_x = np.clip(self.position_x, 0.02, 0.98)
            anchor_y = np.clip(self.position_y, 0.02, 0.98)
            
            # Clear existing actors
            self._clear_actors(plotter)

            renderer = plotter.renderer
            if renderer is None:
                return

            display_height = window_size[1]
            default_start_x = anchor_x * display_width
            default_start_y = anchor_y * display_height

            def _project_reference_bbox() -> Optional[Tuple[float, float, float, float]]:
                """Project reference actors' bounds to display to place bar just below them."""
                if self._reference_actor_fn is None:
                    return None
                actors = []
                try:
                    actors = self._reference_actor_fn(plotter) or []
                except Exception:
                    actors = []
                if not actors:
                    return None
                display_pts: List[Tuple[float, float]] = []
                for actor in actors:
                    try:
                        bounds = actor.GetBounds()
                        if not bounds or len(bounds) < 6:
                            continue
                        xmin, xmax, ymin, ymax, zmin, zmax = bounds
                        corners = [
                            (xmin, ymin, zmin), (xmin, ymin, zmax),
                            (xmin, ymax, zmin), (xmin, ymax, zmax),
                            (xmax, ymin, zmin), (xmax, ymin, zmax),
                            (xmax, ymax, zmin), (xmax, ymax, zmax),
                        ]
                        for cx, cy, cz in corners:
                            renderer.SetWorldPoint(cx, cy, cz, 1.0)
                            renderer.WorldToDisplay()
                            dp = renderer.GetDisplayPoint()
                            display_pts.append((float(dp[0]), float(dp[1])))
                    except Exception:
                        continue
                if not display_pts:
                    return None
                xs = [p[0] for p in display_pts]
                ys = [p[1] for p in display_pts]
                return min(xs), max(xs), min(ys), max(ys)

            ref_bbox = _project_reference_bbox()
            if self._user_anchor is not None:
                start_display_x = np.clip(self._user_anchor[0] * display_width, display_width * 0.02, display_width * 0.98 - pixel_length)
                start_display_y = np.clip(self._user_anchor[1] * display_height, display_height * 0.02, display_height * 0.98)
            elif ref_bbox:
                min_x, max_x, _min_y, max_y = ref_bbox
                margin_px = 12.0
                start_display_x = np.clip(max_x - pixel_length, display_width * 0.02, display_width * 0.98 - pixel_length)
                start_display_y = np.clip(max_y + margin_px, display_height * 0.02, display_height * 0.98)
            else:
                start_display_x = default_start_x
                start_display_y = default_start_y

            end_display_x = start_display_x + pixel_length

            color_rgb = mcolors.to_rgb(self.color)
            text_rgb = mcolors.to_rgb(self.text_color)

            def _add_polyline_actor(points: vtkPoints, cells: vtkCellArray, line_width: float) -> vtkActor2D:
                poly = vtkPolyData()
                poly.SetPoints(points)
                poly.SetLines(cells)
                mapper = vtkPolyDataMapper2D()
                mapper.SetInputData(poly)
                actor2d = vtkActor2D()
                actor2d.SetMapper(mapper)
                actor2d.GetPositionCoordinate().SetCoordinateSystemToDisplay()
                actor2d.GetProperty().SetColor(*color_rgb)
                actor2d.GetProperty().SetLineWidth(max(1.0, float(line_width)))
                renderer.AddActor2D(actor2d)
                return actor2d

            def _line_to_cells(points_arr: List[Tuple[float, float]]) -> Tuple[vtkPoints, vtkCellArray]:
                pts = vtkPoints()
                cells = vtkCellArray()
                for p1, p2 in points_arr:
                    id1 = pts.InsertNextPoint(p1[0], p1[1], 0.0)
                    id2 = pts.InsertNextPoint(p2[0], p2[1], 0.0)
                    line = vtkLine()
                    line.GetPointIds().SetId(0, id1)
                    line.GetPointIds().SetId(1, id2)
                    cells.InsertNextCell(line)
                return pts, cells

            # Main bar as screen-space polyline
            bar_pts, bar_cells = _line_to_cells([([start_display_x, start_display_y], [end_display_x, start_display_y])])
            self.bar_actor = _add_polyline_actor(bar_pts, bar_cells, self.line_width)
            
            # Determine tick mark sizes (as fraction of view height)
            major_tick_length_px = max(8.0, window_size[1] * 0.02)  # 2% of view height
            minor_tick_length_px = max(4.0, window_size[1] * 0.01)  # 1% of view height
            
            # Calculate major tick interval in DISPLAY units
            # Aim for approximately 6-8 major ticks across the scale bar
            target_num_ticks = 7
            display_length = self._current_display_length
            major_interval = display_length / max(1, target_num_ticks - 1)

            # Round to a nice number (1, 2, 5, or 10 times power of 10) in display units
            if major_interval > 0:
                exp = int(np.floor(np.log10(major_interval)))
                base = 10 ** exp
                candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
                major_interval = candidates[np.abs(candidates - major_interval).argmin()]

            # Calculate number of major ticks based on rounded interval
            num_major_ticks = int(np.ceil(display_length / major_interval)) + 1 if major_interval > 0 else 2

            # Determine minor tick interval (divide major interval by 10)
            minor_interval = major_interval / 10.0
            
            # Determine text height
            font_size_px = int(max(10.0, self.text_height if self.text_height is not None else window_size[1] * 0.018))
            
            # Create tick marks and labels
            label_offset_px = major_tick_length_px * 1.7
            tick_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            minor_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            min_label_spacing_px = max(font_size_px * 2.2, pixel_length / 6.0)
            last_label_x = None

            for i in range(num_major_ticks):
                tick_value = i * major_interval  # In display units
                if tick_value > display_length + 1e-6:
                    break

                frac = tick_value / max(display_length, 1e-6)
                tick_display_x = start_display_x + frac * pixel_length

                tick_segments.append(
                    ([tick_display_x, start_display_y - major_tick_length_px * 0.5],
                     [tick_display_x, start_display_y + major_tick_length_px * 0.5])
                )

                should_force_label = (
                    i == 0
                    or i == num_major_ticks - 1
                    or abs(tick_value - display_length) < 1e-6
                )
                allow_label = False
                if last_label_x is None:
                    allow_label = True
                elif abs(tick_display_x - last_label_x) >= min_label_spacing_px or should_force_label:
                    allow_label = True

                if allow_label:
                    if abs(tick_value - round(tick_value)) < 1e-6:
                        label_text = f"{int(tick_value)}"
                    else:
                        label_text = f"{tick_value:.1f}"

                    label_actor = vtkTextActor()
                    label_actor.SetInput(label_text)
                    label_prop = label_actor.GetTextProperty()
                    label_prop.SetFontFamilyToArial()
                    label_prop.SetFontSize(font_size_px)
                    label_prop.SetColor(*text_rgb)
                    label_prop.SetJustificationToCentered()
                    label_prop.SetVerticalJustificationToTop()
                    label_actor.SetDisplayPosition(int(tick_display_x), int(start_display_y - label_offset_px))
                    renderer.AddActor2D(label_actor)
                    self.label_actors.append(label_actor)
                    last_label_x = tick_display_x

                # Minor ticks
                if i < num_major_ticks - 1:
                    for j in range(1, 10):
                        minor_value = tick_value + j * minor_interval
                        if minor_value >= (i + 1) * major_interval:
                            break

                        minor_frac = minor_value / max(display_length, 1e-6)
                        minor_display_x = start_display_x + minor_frac * pixel_length
                        minor_segments.append(
                            ([minor_display_x, start_display_y - minor_tick_length_px * 0.5],
                             [minor_display_x, start_display_y + minor_tick_length_px * 0.5])
                        )

            # Create tick actors in screen space
            if tick_segments:
                pts, cells = _line_to_cells(tick_segments)
                actor = _add_polyline_actor(pts, cells, self.line_width)
                self.major_tick_actors.append(actor)

            if minor_segments:
                pts, cells = _line_to_cells(minor_segments)
                actor = _add_polyline_actor(pts, cells, max(1.0, self.line_width * 0.6))
                self.minor_tick_actors.append(actor)
            
            # Add unit label at the end (display coordinates below scale bar)
            unit_label_display_x = end_display_x
            unit_label_display_y = start_display_y - label_offset_px * 2

            unit_display = self.units
            if unit_display.lower() == "m":
                unit_display = "Meters"
            elif unit_display.lower() == "km":
                unit_display = "Kilometers"
            elif unit_display.lower() == "ft":
                unit_display = "Feet"
            elif unit_display.lower() == "mm":
                unit_display = "Millimeters"

            unit_label_actor = vtkTextActor()
            unit_label_actor.SetInput(unit_display)
            unit_prop = unit_label_actor.GetTextProperty()
            unit_prop.SetFontFamilyToArial()
            unit_prop.SetFontSize(font_size_px)
            unit_prop.SetColor(*text_rgb)
            unit_prop.SetJustificationToRight()
            unit_prop.SetVerticalJustificationToTop()
            unit_label_actor.SetDisplayPosition(int(unit_label_display_x), int(unit_label_display_y))
            renderer.AddActor2D(unit_label_actor)
            self.unit_label_actor = unit_label_actor

            self._last_display_origin = (start_display_x, start_display_y)
            self._last_pixel_length = pixel_length
            self._last_bar_bounds = (
                start_display_x - 10.0,
                start_display_y - major_tick_length_px * 1.5 - label_offset_px * 2,
                end_display_x + 10.0,
                start_display_y + major_tick_length_px * 1.5,
            )
        
        except Exception as e:
            logger.error(f"ScaleBar3D.update failed: {e}", exc_info=True)
    
    def _clear_actors(self, plotter: pv.Plotter) -> None:
        """Remove all scale bar actors from the plotter."""
        def _remove_actor_safe(actor_obj):
            if actor_obj is None:
                return
            try:
                plotter.remove_actor(actor_obj)
            except Exception:
                try:
                    if hasattr(plotter, "renderer") and plotter.renderer is not None:
                        try:
                            plotter.renderer.RemoveActor(actor_obj)
                        except Exception:
                            plotter.renderer.RemoveActor2D(actor_obj)
                except Exception:
                    pass

        if self.bar_actor is not None:
            _remove_actor_safe(self.bar_actor)
            self.bar_actor = None
        
        for actor in self.major_tick_actors:
            _remove_actor_safe(actor)
        self.major_tick_actors.clear()
        
        for actor in self.minor_tick_actors:
            _remove_actor_safe(actor)
        self.minor_tick_actors.clear()
        
        for actor in self.label_actors:
            _remove_actor_safe(actor)
        self.label_actors.clear()
        
        if self.unit_label_actor is not None:
            _remove_actor_safe(self.unit_label_actor)
            self.unit_label_actor = None
        self._last_bar_bounds = None

    def clear(self, plotter: pv.Plotter) -> None:
        """Remove scale bar actors from the plotter."""
        self._clear_actors(plotter)
        # Remove camera observer to prevent memory leaks
        if self._camera_observer_tag is not None:
            try:
                camera = plotter.camera
                if camera is not None:
                    camera.RemoveObserver(self._camera_observer_tag)
            except Exception:
                pass
            self._camera_observer_tag = None
        self._callback_registered = False
        self._interactor_observers_registered = False

    def attach_plotter(self, plotter: pv.Plotter) -> None:
        """Store a plotter reference for interactive repositioning."""
        self._plotter_ref = plotter
        self._install_interaction(plotter)

    def _install_interaction(self, plotter: pv.Plotter) -> None:
        """Enable drag interactions for the scale bar."""
        if self._interaction_installed:
            return
        interactor = getattr(plotter, 'iren', None) or getattr(plotter, 'interactor', None)
        if interactor is None:
            logger.debug("No interactor found for scale bar drag interaction")
            return
        self._interactor = interactor

        # Try both PyVistaQt (lowercase) and VTK (CamelCase) method names
        try:
            add_obs = getattr(interactor, 'add_observer', None) or getattr(interactor, 'AddObserver', None)
            if add_obs is None:
                logger.debug("No add_observer method found on interactor")
                return
            self._press_tag = add_obs("LeftButtonPressEvent", self._on_left_press, 1.0)
            self._move_tag = add_obs("MouseMoveEvent", self._on_mouse_move, 1.0)
            self._release_tag = add_obs("LeftButtonReleaseEvent", self._on_left_release, 1.0)
            self._interaction_installed = True
            logger.debug("Scale bar drag interaction installed")
        except Exception as e:
            logger.debug(f"Could not install scale bar drag interaction: {e}")

    def _within_bar(self, x: float, y: float) -> bool:
        if self._last_bar_bounds is None:
            return False
        x0, y0, x1, y1 = self._last_bar_bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    def _on_left_press(self, obj, event) -> None:
        if not self._plotter_ref:
            return
        x, y = obj.GetEventPosition()
        if not self._within_bar(x, y):
            return
        if self._last_display_origin is None:
            return
        self._dragging = True
        self._drag_offset = (x - self._last_display_origin[0], y - self._last_display_origin[1])
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _on_mouse_move(self, obj, event) -> None:
        if not self._dragging or not self._plotter_ref:
            return
        plotter = self._plotter_ref
        x, y = obj.GetEventPosition()
        width, height = plotter.window_size
        if width <= 0 or height <= 0:
            return
        start_x = x - self._drag_offset[0]
        start_y = y - self._drag_offset[1]
        margin = 12.0
        start_x = np.clip(start_x, margin, width - margin - max(self._last_pixel_length, 1.0))
        start_y = np.clip(start_y, margin, height - margin)
        self._user_anchor = (start_x / width, start_y / height)
        self.position_x, self.position_y = self._user_anchor
        self.update(plotter)
        try:
            plotter.render()
        except Exception:
            pass
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _on_left_release(self, obj, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def get_actors(self) -> List[pv.Actor]:
        """Return all PyVista actors that belong to the scale bar."""
        actors = []
        if self.bar_actor is not None:
            actors.append(self.bar_actor)
        actors.extend(actor for actor in self.major_tick_actors if actor is not None)
        actors.extend(actor for actor in self.minor_tick_actors if actor is not None)
        actors.extend(actor for actor in self.label_actors if actor is not None)
        if self.unit_label_actor is not None:
            actors.append(self.unit_label_actor)
        return actors

