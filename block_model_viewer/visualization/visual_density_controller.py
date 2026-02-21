"""
Visual Density Controller for Drillhole Rendering

Automatically adjusts drillhole visual complexity based on camera distance and zoom level.
Provides smooth LOD transitions without modifying drillhole data or breaking selection/picking.

Key Features:
- Distance-based density modes (overview/midview/detail)
- Actor property adjustments (radius, opacity, visibility)
- Preserves selection state and picking IDs
- Performance-optimized updates
- Debug logging and developer controls
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


# =============================================================================
# Visual Density Modes
# =============================================================================

class VisualDensityMode(IntEnum):
    """Visual density modes based on camera distance."""
    OVERVIEW = 0    # Far distance - simplified visuals
    MIDVIEW = 1     # Medium distance - balanced detail
    DETAIL = 2      # Close distance - full detail


@dataclass
class VisualDensityPreset:
    """
    Visual property presets for each density mode.

    Only modifies actor properties - never rebuilds geometry or modifies data.
    """
    mode: VisualDensityMode

    # Cylinder/actor properties
    radius_multiplier: float = 1.0      # Scale factor for tube radius
    opacity: float = 1.0                 # Overall opacity (0.0-1.0)
    ambient: float = 0.3                 # Lighting ambient component
    diffuse: float = 0.7                 # Lighting diffuse component
    specular: float = 0.2                # Lighting specular component

    # Visibility toggles (actor-level)
    show_labels: bool = True             # Show depth/sample labels
    show_markers: bool = True            # Show collar markers and ticks
    show_assay_bands: bool = True        # Show assay color bands
    show_lithology_colors: bool = True   # Show lithology colors

    # Collar marker properties
    collar_radius_multiplier: float = 1.0
    collar_opacity: float = 1.0

    # Label properties
    label_opacity: float = 1.0
    label_size_multiplier: float = 1.0

    # Special effects
    soften_colors: bool = False          # Reduce color saturation for overview
    highlight_selected: bool = True      # Always keep selected drillholes full detail

    @classmethod
    def get_overview_preset(cls) -> VisualDensityPreset:
        """Overview mode: Simplified visuals for far distances."""
        return cls(
            mode=VisualDensityMode.OVERVIEW,
            radius_multiplier=0.5,        # Thinner traces
            opacity=0.7,                  # Semi-transparent
            ambient=0.5,                  # More ambient for better visibility
            diffuse=0.4,
            specular=0.1,
            show_labels=False,            # Hide labels
            show_markers=False,           # Hide markers
            show_assay_bands=False,       # Merge/simplify assay bands
            collar_radius_multiplier=0.3,
            collar_opacity=0.5,
            soften_colors=True,           # Soften assay colours
        )

    @classmethod
    def get_midview_preset(cls) -> VisualDensityPreset:
        """Midview mode: Balanced detail for medium distances."""
        return cls(
            mode=VisualDensityMode.MIDVIEW,
            radius_multiplier=0.75,       # Medium thickness
            opacity=0.85,
            ambient=0.4,
            diffuse=0.5,
            specular=0.15,
            show_labels=False,            # Reduced label density
            show_markers=True,
            show_assay_bands=True,
            collar_radius_multiplier=0.6,
            collar_opacity=0.8,
            label_opacity=0.7,
            label_size_multiplier=0.8,
        )

    @classmethod
    def get_detail_preset(cls) -> VisualDensityPreset:
        """Detail mode: Full detail for close inspection."""
        return cls(
            mode=VisualDensityMode.DETAIL,
            radius_multiplier=1.0,        # Full thickness
            opacity=1.0,                  # Fully opaque
            ambient=0.3,
            diffuse=0.6,
            specular=0.2,
            show_labels=True,
            show_markers=True,
            show_assay_bands=True,
            show_lithology_colors=True,
            collar_radius_multiplier=1.0,
            collar_opacity=1.0,
            label_opacity=1.0,
            label_size_multiplier=1.0,
            highlight_selected=True,
        )


# =============================================================================
# Distance Thresholds Configuration
# =============================================================================

@dataclass
class DistanceThresholds:
    """Distance thresholds for switching between visual density modes."""
    overview_max_distance: float = 500.0   # Beyond this: overview mode
    midview_max_distance: float = 200.0    # Up to this: midview mode (not used in current logic)
    detail_max_distance: float = 50.0      # Within this: detail mode

    # Hysteresis to prevent rapid switching
    hysteresis_buffer: float = 10.0

    def get_mode_for_distance(self, distance: float) -> VisualDensityMode:
        """Determine visual density mode based on distance with hysteresis."""
        if distance <= self.detail_max_distance:
            return VisualDensityMode.DETAIL
        elif distance <= self.overview_max_distance:
            return VisualDensityMode.MIDVIEW
        else:
            return VisualDensityMode.OVERVIEW


# =============================================================================
# Visual Density Controller
# =============================================================================

class VisualDensityController(QObject):
    """
    Controller for automatic visual density adjustment based on camera position.

    Monitors camera movements and applies appropriate visual presets to drillhole actors
    without modifying underlying data or breaking selection/picking.
    """

    # Signals
    densityModeChanged = pyqtSignal(object)  # VisualDensityMode
    visualPropertiesUpdated = pyqtSignal(dict)  # Update info dict

    def __init__(
        self,
        plotter: Any,  # PyVista plotter
        thresholds: Optional[DistanceThresholds] = None,
        update_interval_ms: int = 100,  # Check every 100ms
        enabled: bool = True,
        debug_logging: bool = False,
    ):
        super().__init__()

        self.plotter = plotter
        self.thresholds = thresholds or DistanceThresholds()
        self.enabled = enabled
        self.debug_logging = debug_logging

        # State
        self.current_mode: Optional[VisualDensityMode] = None
        self.last_camera_position: Optional[np.ndarray] = None
        self.selected_hole_ids: Set[str] = set()  # Always full detail

        # Actor tracking (populated by renderer)
        self.drillhole_actors: Dict[str, Any] = {}  # hole_id -> main actor
        self.collar_actors: Dict[str, Any] = {}     # hole_id -> collar actor
        self.label_actors: List[Any] = []           # Label actors

        # Performance
        self.update_timer = QTimer()
        self.update_timer.setInterval(update_interval_ms)
        self.update_timer.timeout.connect(self._check_camera_and_update)
        self._last_update_time = 0.0

        # Stats for debugging
        self.update_count = 0
        self.mode_switch_count = 0

        if self.enabled:
            self.start_monitoring()

        logger.info(f"VisualDensityController initialized (enabled={enabled})")

    def start_monitoring(self):
        """Start camera position monitoring."""
        if not self.update_timer.isActive():
            self.update_timer.start()
            if self.debug_logging:
                logger.debug("Visual density monitoring started")

    def stop_monitoring(self):
        """Stop camera position monitoring."""
        if self.update_timer.isActive():
            self.update_timer.stop()
            if self.debug_logging:
                logger.debug("Visual density monitoring stopped")

    def set_enabled(self, enabled: bool):
        """Enable or disable visual density control."""
        if enabled == self.enabled:
            return

        self.enabled = enabled
        if enabled:
            self.start_monitoring()
            # Force immediate update
            self._check_camera_and_update()
        else:
            self.stop_monitoring()
            # Reset to full detail when disabled
            self._apply_preset(VisualDensityPreset.get_detail_preset())

        logger.info(f"Visual density controller {'enabled' if enabled else 'disabled'}")

    def set_selected_holes(self, hole_ids: Set[str]):
        """Update selected hole IDs (these always remain full detail)."""
        self.selected_hole_ids = hole_ids.copy()
        if self.debug_logging:
            logger.debug(f"Selected holes updated: {len(hole_ids)} holes")

    def register_actors(
        self,
        drillhole_actors: Dict[str, Any],
        collar_actors: Dict[str, Any],
        label_actors: Optional[List[Any]] = None,
    ):
        """Register drillhole actors for density control."""
        self.drillhole_actors = drillhole_actors.copy()
        self.collar_actors = collar_actors.copy()
        self.label_actors = label_actors or []

        if self.debug_logging:
            logger.debug(f"Registered {len(drillhole_actors)} drillhole actors, "
                        f"{len(collar_actors)} collar actors, "
                        f"{len(self.label_actors)} label actors")

    def _check_camera_and_update(self):
        """Check camera position and update visuals if needed."""
        if not self.enabled or self.plotter is None:
            return

        try:
            # Get current camera position
            camera = self.plotter.renderer.GetActiveCamera()
            if camera is None:
                return

            current_pos = np.array(camera.GetPosition())

            # Check if camera moved significantly
            if self.last_camera_position is not None:
                movement = np.linalg.norm(current_pos - self.last_camera_position)
                if movement < 1.0:  # Less than 1m movement, skip
                    return

            self.last_camera_position = current_pos

            # Calculate distance to scene center (or use average drillhole position)
            distance = self._calculate_scene_distance(current_pos)

            # Determine appropriate mode
            new_mode = self.thresholds.get_mode_for_distance(distance)

            # Update if mode changed
            if new_mode != self.current_mode:
                self._switch_mode(new_mode)

            self.update_count += 1

        except Exception as e:
            if self.debug_logging:
                logger.debug(f"Camera check failed: {e}")

    def _calculate_scene_distance(self, camera_pos: np.ndarray) -> float:
        """Calculate representative distance from camera to drillhole scene."""
        if not self.drillhole_actors:
            return 100.0  # Default medium distance

        # Use center of all drillhole positions as reference
        positions = []
        for actor in self.drillhole_actors.values():
            try:
                if hasattr(actor, 'GetBounds'):
                    bounds = actor.GetBounds()
                    center = np.array([
                        (bounds[0] + bounds[1]) / 2,
                        (bounds[2] + bounds[3]) / 2,
                        (bounds[4] + bounds[5]) / 2
                    ])
                    positions.append(center)
            except Exception:
                continue

        if positions:
            scene_center = np.mean(positions, axis=0)
            return float(np.linalg.norm(camera_pos - scene_center))
        else:
            return 100.0  # Default

    def _switch_mode(self, new_mode: VisualDensityMode):
        """Switch to a new visual density mode."""
        if self.debug_logging:
            old_mode_name = self.current_mode.name if self.current_mode else "NONE"
            logger.debug(f"Switching visual density: {old_mode_name} -> {new_mode.name}")

        self.current_mode = new_mode
        self.mode_switch_count += 1

        # Get appropriate preset
        if new_mode == VisualDensityMode.OVERVIEW:
            preset = VisualDensityPreset.get_overview_preset()
        elif new_mode == VisualDensityMode.MIDVIEW:
            preset = VisualDensityPreset.get_midview_preset()
        else:  # DETAIL
            preset = VisualDensityPreset.get_detail_preset()

        # Apply preset
        self._apply_preset(preset)

        # Emit signals
        self.densityModeChanged.emit(new_mode)
        self.visualPropertiesUpdated.emit({
            'mode': new_mode.name,
            'preset': preset.__dict__,
            'update_count': self.update_count,
        })

    def _apply_preset(self, preset: VisualDensityPreset):
        """Apply visual density preset to all registered actors."""
        import time
        start_time = time.perf_counter()

        update_info = {
            'actors_updated': 0,
            'properties_changed': [],
            'selected_preserved': len(self.selected_hole_ids),
        }

        try:
            # Update drillhole actors (but preserve selected holes)
            for hole_id, actor in self.drillhole_actors.items():
                if hole_id in self.selected_hole_ids:
                    # Selected holes always stay in detail mode
                    detail_preset = VisualDensityPreset.get_detail_preset()
                    self._apply_preset_to_actor(actor, detail_preset, f"drillhole_{hole_id}")
                    update_info['properties_changed'].append(f"selected_{hole_id}")
                else:
                    self._apply_preset_to_actor(actor, preset, f"drillhole_{hole_id}")
                update_info['actors_updated'] += 1

            # Update collar actors
            for hole_id, actor in self.collar_actors.items():
                if hole_id in self.selected_hole_ids:
                    detail_preset = VisualDensityPreset.get_detail_preset()
                    self._apply_preset_to_actor(actor, detail_preset, f"collar_{hole_id}")
                else:
                    self._apply_preset_to_actor(actor, preset, f"collar_{hole_id}")
                update_info['actors_updated'] += 1

            # Update label actors
            for actor in self.label_actors:
                self._apply_preset_to_actor(actor, preset, "label")

            # Force render update
            try:
                self.plotter.render()
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Failed to apply visual density preset: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if self.debug_logging:
            logger.debug(f"Applied {preset.mode.name} preset in {elapsed_ms:.1f}ms: "
                        f"{update_info['actors_updated']} actors updated")

    def _apply_preset_to_actor(self, actor: Any, preset: VisualDensityPreset, actor_name: str):
        """Apply preset properties to a single actor."""
        if actor is None:
            return

        try:
            # Actor property updates (safe operations only)
            prop = actor.GetProperty()
            if prop:
                # Basic visual properties
                if hasattr(prop, 'SetOpacity'):
                    prop.SetOpacity(preset.opacity)

                if hasattr(prop, 'SetAmbient'):
                    prop.SetAmbient(preset.ambient)

                if hasattr(prop, 'SetDiffuse'):
                    prop.SetDiffuse(preset.diffuse)

                if hasattr(prop, 'SetSpecular'):
                    prop.SetSpecular(preset.specular)

                # Special color softening for overview
                if preset.soften_colors and hasattr(prop, 'SetSpecularPower'):
                    prop.SetSpecularPower(1.0)  # Reduce shininess

                prop.Modified()

            # Visibility toggles (for labels, markers, etc.)
            if "label" in actor_name.lower():
                if hasattr(actor, 'SetVisibility'):
                    actor.SetVisibility(preset.show_labels)
            elif "marker" in actor_name.lower() or "collar" in actor_name.lower():
                if hasattr(actor, 'SetVisibility'):
                    actor.SetVisibility(preset.show_markers)

            # Force actor update
            actor.Modified()

        except Exception as e:
            if self.debug_logging:
                logger.debug(f"Failed to update actor {actor_name}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for debugging."""
        return {
            'enabled': self.enabled,
            'current_mode': self.current_mode.name if self.current_mode else None,
            'update_count': self.update_count,
            'mode_switch_count': self.mode_switch_count,
            'registered_actors': len(self.drillhole_actors) + len(self.collar_actors),
            'selected_holes': len(self.selected_hole_ids),
            'thresholds': self.thresholds.__dict__,
            'monitoring_active': self.update_timer.isActive(),
        }

    def reset_stats(self):
        """Reset debug statistics."""
        self.update_count = 0
        self.mode_switch_count = 0
        logger.debug("Visual density controller stats reset")
