"""
Visualization Controller - Handles all rendering and display operations.

This controller manages colormap, transparency, layers, legend, overlays,
scene manipulation, styling, and screenshot export.
"""

from typing import Optional, Dict, Any, Callable, Tuple, Sequence, Union, TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    from .app_controller import AppController

logger = logging.getLogger(__name__)


class VisController:
    """
    Controller for visualization and rendering operations.
    
    Handles colormap/transparency, layer management, legend settings,
    overlay visibility, scene controls, styling, and screenshot export.
    """
    
    def __init__(self, app_controller: "AppController"):
        """
        Initialize visualization controller.
        
        Args:
            app_controller: Parent AppController instance for shared state access
        """
        self._app = app_controller
    
    @property
    def renderer(self):
        """Return the renderer instance."""
        return self._app.r
    
    @property
    def session(self):
        """Return the session state."""
        return self._app.s
    
    @property
    def legend_manager(self):
        """Return the legend manager."""
        return self._app.legend_manager
    
    @property
    def overlay_manager(self):
        """Return the overlay manager."""
        return self._app.overlay_manager
    
    @property
    def axis_manager(self):
        """Return the axis manager."""
        return self._app.axis_manager
    
    # =========================================================================
    # Colormap and Transparency
    # =========================================================================
    
    def set_colormap(self, cmap: str) -> None:
        """
        Set the colormap for visualization.
        
        Args:
            cmap: Name of the colormap (e.g., "viridis", "plasma", "jet")
        """
        self.session.color_map = cmap
        try:
            if hasattr(self.renderer, "set_colormap"):
                self.renderer.set_colormap(cmap)
            elif hasattr(self.renderer, "colormap"):
                self.renderer.colormap = cmap
            
            # Update legend if available
            if self.legend_manager:
                try:
                    self.legend_manager.set_colormap(cmap)
                except Exception:
                    logger.debug("Legend manager colormap update failed", exc_info=True)
            
            logger.debug(f"Colormap set to {cmap}")
        except Exception as e:
            logger.warning(f"Failed to set colormap: {e}")
    
    def set_transparency(self, alpha: float) -> None:
        """
        Set global transparency for visualization.
        
        Args:
            alpha: Transparency value (0.0 = fully transparent, 1.0 = fully opaque)
        """
        self.session.transparency = float(alpha)
        if hasattr(self.renderer, "set_transparency"):
            try:
                self.renderer.set_transparency(float(alpha))
            except Exception as e:
                logger.warning(f"Failed to set transparency: {e}")
    
    def set_global_opacity(self, opacity: float) -> None:
        """
        Set global opacity for all layers.
        
        Args:
            opacity: Opacity value (0.0 = transparent, 1.0 = opaque)
        """
        if hasattr(self.renderer, "set_transparency"):
            try:
                self.renderer.set_transparency(float(opacity))
            except Exception:
                logger.debug("Renderer set_transparency failed", exc_info=True)
    
    # =========================================================================
    # Active Property
    # =========================================================================
    
    def set_active_property(self, name: Optional[str]) -> None:
        """
        Set the active property for visualization.
        
        Args:
            name: Property name to display, or None to clear
        """
        self.session.current_property = name
        if hasattr(self.renderer, "set_active_property"):
            try:
                self.renderer.set_active_property(name)
            except Exception as e:
                logger.warning(f"Failed to set active property: {e}")
        
        # Update legend
        if self.legend_manager and name:
            try:
                self.legend_manager.update_from_property(name, {"colormap": self.session.color_map})
            except Exception:
                logger.debug("Legend update failed", exc_info=True)
        
        self._apply_visual_prefs()
    
    def set_current_property(self, name: Optional[str]) -> None:
        """
        Alias for set_active_property.
        
        Args:
            name: Property name to display, or None to clear
        """
        self.set_active_property(name)
    
    # =========================================================================
    # Legend Management
    # =========================================================================
    
    def toggle_legend(self, show: bool) -> None:
        """
        Toggle legend visibility.
        
        Args:
            show: Whether to show the legend
        """
        self.session.legend_visible = bool(show)
        
        if self.legend_manager:
            try:
                self.legend_manager.set_visibility(bool(show))
            except Exception:
                logger.debug("Legend manager visibility toggle failed", exc_info=True)
        
        if hasattr(self.renderer, "toggle_legend"):
            try:
                self.renderer.toggle_legend(bool(show))
            except Exception:
                logger.debug("Renderer toggle_legend failed", exc_info=True)
        elif hasattr(self.renderer, "toggle_scalar_bar"):
            try:
                self.renderer.toggle_scalar_bar(bool(show))
            except Exception:
                logger.debug("Renderer toggle_scalar_bar failed", exc_info=True)
    
    def set_legend_visibility(self, visible: bool) -> None:
        """
        Set legend visibility.
        
        Args:
            visible: Whether to show the legend
        """
        if self.legend_manager is not None:
            try:
                self.legend_manager.set_visibility(bool(visible))
            except Exception:
                logger.debug("Legend manager visibility failed", exc_info=True)
        self.toggle_legend(bool(visible))
    
    def set_legend_orientation(self, orientation: str) -> None:
        """
        Set legend orientation.
        
        Args:
            orientation: Orientation ("horizontal" or "vertical")
        """
        legend = getattr(self.renderer, "legend_manager", None)
        if legend is not None and hasattr(legend, "set_orientation"):
            try:
                legend.set_orientation(orientation)
            except Exception:
                logger.debug("Legend manager orientation update failed", exc_info=True)
        elif hasattr(self.renderer, "set_legend_orientation"):
            try:
                self.renderer.set_legend_orientation(orientation)
            except Exception:
                logger.debug("Renderer legend orientation update failed", exc_info=True)
    
    def set_legend_font_size(self, size: int) -> None:
        """
        Set legend font size.
        
        Args:
            size: Font size in points
        """
        legend = getattr(self.renderer, "legend_manager", None)
        if legend and hasattr(legend, "set_font_size"):
            try:
                legend.set_font_size(int(size))
            except Exception:
                logger.debug("Legend manager font size update failed", exc_info=True)
        elif hasattr(self.renderer, "legend_widget") and getattr(self.renderer, "legend_widget"):
            try:
                self.renderer.legend_widget.config.font_size = int(size)
                self.renderer.legend_widget.update()
            except Exception:
                logger.debug("Legend widget font size update failed", exc_info=True)
    
    def reset_legend_position(self) -> None:
        """Reset legend to default position."""
        if hasattr(self.renderer, "reset_legend_position"):
            try:
                self.renderer.reset_legend_position()
            except Exception:
                logger.debug("Renderer reset_legend_position failed", exc_info=True)
    
    # =========================================================================
    # Layer Management
    # =========================================================================
    
    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """
        Set visibility of a specific layer.
        
        Args:
            layer_name: Name of the layer
            visible: Whether the layer should be visible
        """
        try:
            if hasattr(self.renderer, "set_layer_visibility"):
                self.renderer.set_layer_visibility(layer_name, bool(visible))
        finally:
            self._app.notify_panels_scene_changed()
    
    def set_layer_opacity(self, layer_name: str, opacity: float) -> None:
        """
        Set opacity of a specific layer.
        
        Args:
            layer_name: Name of the layer
            opacity: Opacity value (0.0-1.0)
        """
        try:
            if hasattr(self.renderer, "set_layer_opacity"):
                self.renderer.set_layer_opacity(layer_name, float(opacity))
        finally:
            self._app.notify_panels_scene_changed()
    
    def set_active_layer(self, layer_name: str) -> None:
        """
        Set the active layer for controls.
        
        Args:
            layer_name: Name of the layer to activate
        """
        if hasattr(self.renderer, "set_active_layer_for_controls"):
            try:
                self.renderer.set_active_layer_for_controls(layer_name)
            except Exception as exc:
                logger.error("Failed to set active layer '%s': %s", layer_name, exc, exc_info=True)
    
    def remove_layer(self, layer_name: str) -> None:
        """
        Remove a layer from the scene.
        
        Args:
            layer_name: Name of the layer to remove
        """
        try:
            if hasattr(self.renderer, "clear_layer"):
                self.renderer.clear_layer(layer_name)
        finally:
            self._app.notify_panels_scene_changed()
    
    # =========================================================================
    # Overlay Management
    # =========================================================================
    
    def set_axes_visible(self, show: bool) -> None:
        """
        Set axes visibility.
        
        Args:
            show: Whether to show axes
        """
        if hasattr(self.renderer, "toggle_axes"):
            try:
                self.renderer.toggle_axes(bool(show))
            except Exception:
                logger.debug("Renderer toggle_axes failed", exc_info=True)
        if self.overlay_manager is not None:
            self.overlay_manager.toggle_overlay("axes", bool(show))
    
    def set_bounds_visible(self, show: bool) -> None:
        """
        Set bounds visibility.
        
        Args:
            show: Whether to show bounding box
        """
        if hasattr(self.renderer, "toggle_bounds"):
            try:
                self.renderer.toggle_bounds(bool(show))
            except Exception:
                logger.debug("Renderer toggle_bounds failed", exc_info=True)
        if self.overlay_manager is not None:
            self.overlay_manager.toggle_overlay("bounds", bool(show))
    
    def set_ground_grid_visible(self, show: bool) -> None:
        """
        Set ground grid visibility.
        
        Args:
            show: Whether to show ground grid
        """
        if hasattr(self.renderer, "set_show_ground_grid"):
            try:
                self.renderer.set_show_ground_grid(bool(show))
            except Exception:
                logger.debug("Renderer set_show_ground_grid failed", exc_info=True)
        if self.overlay_manager is not None:
            self.overlay_manager.toggle_overlay("ground_grid", bool(show))
    
    def set_ground_grid_spacing(self, spacing: float) -> None:
        """
        Set ground grid spacing.
        
        Args:
            spacing: Grid spacing in model units
        """
        if hasattr(self.renderer, "set_ground_grid_spacing"):
            try:
                self.renderer.set_ground_grid_spacing(float(spacing))
            except Exception:
                logger.debug("Renderer set_ground_grid_spacing failed", exc_info=True)
    
    def reset_ground_grid_spacing(self) -> None:
        """Reset ground grid to default spacing."""
        if hasattr(self.renderer, "reset_ground_grid_spacing"):
            try:
                self.renderer.reset_ground_grid_spacing()
            except Exception:
                logger.debug("Renderer reset_ground_grid_spacing failed", exc_info=True)
    
    def set_overlay_units(self, units: str) -> None:
        """
        Set overlay units (e.g., "m", "ft").
        
        Args:
            units: Unit string
        """
        if hasattr(self.renderer, "set_overlay_units"):
            try:
                self.renderer.set_overlay_units(units)
            except Exception:
                logger.debug("Renderer set_overlay_units failed", exc_info=True)
    
    def configure_overlays(self, **kwargs) -> None:
        """
        Forward overlay configuration updates to the renderer.
        
        Args:
            **kwargs: Overlay name -> visibility pairs
        """
        manager_handled = False
        if self.overlay_manager is not None:
            for name, value in kwargs.items():
                try:
                    self.overlay_manager.toggle_overlay(name, value)
                    manager_handled = True
                except Exception:
                    pass
        if hasattr(self.renderer, "configure_overlays"):
            try:
                self.renderer.configure_overlays(**kwargs)
                return
            except Exception as exc:
                logger.error("Renderer overlay configuration failed: %s", exc, exc_info=True)
        if not manager_handled:
            logger.debug("Renderer does not support configure_overlays; kwargs=%s", kwargs)
    
    # =========================================================================
    # Viewer Styling
    # =========================================================================
    
    def set_background_color(self, color: Tuple[float, float, float]) -> None:
        """
        Set background color.
        
        Args:
            color: RGB tuple (0.0-1.0 for each component)
        """
        if hasattr(self.renderer, "set_background_color"):
            try:
                self.renderer.set_background_color(color)
            except Exception:
                logger.debug("Renderer set_background_color failed", exc_info=True)
    
    def set_edge_color(self, color: Tuple[float, float, float]) -> None:
        """
        Set edge color for meshes.

        Args:
            color: RGB tuple (0.0-1.0 for each component)
        """
        if hasattr(self.renderer, "set_edge_color"):
            try:
                self.renderer.set_edge_color(color)
            except Exception:
                logger.debug("Renderer set_edge_color failed", exc_info=True)

    def set_edge_visibility(self, visible: bool) -> None:
        """
        Set edge visibility for block models.

        Args:
            visible: Whether edges should be visible
        """
        if hasattr(self.renderer, "set_edge_visibility"):
            try:
                self.renderer.set_edge_visibility(visible)
            except Exception:
                logger.debug("Renderer set_edge_visibility failed", exc_info=True)

    def set_lighting_enabled(self, enabled: bool) -> None:
        """
        Enable or disable lighting.
        
        Args:
            enabled: Whether lighting should be enabled
        """
        if hasattr(self.renderer, "set_lighting_enabled"):
            try:
                self.renderer.set_lighting_enabled(bool(enabled))
            except Exception:
                logger.debug("Renderer set_lighting_enabled failed", exc_info=True)
    
    # =========================================================================
    # Scene Controls
    # =========================================================================
    
    def reset_scene(self) -> None:
        """Reset the scene to initial state."""
        if hasattr(self.renderer, "clear_scene"):
            try:
                self.renderer.clear_scene()
            except Exception as e:
                logger.warning(f"Failed to reset scene: {e}")
        self.session.current_property = None
        self.session.filter_specs = {}
        self._app.notify_panels_scene_changed()
    
    def refresh_scene(self) -> None:
        """Refresh the scene visualization."""
        if self.renderer is None:
            logger.debug("No renderer available for refresh_scene")
            return
        
        try:
            if hasattr(self.renderer, "refresh"):
                self.renderer.refresh()
            elif hasattr(self.renderer, "render"):
                self.renderer.render()
            elif hasattr(self.renderer, "update"):
                self.renderer.update()
        except Exception as e:
            logger.warning(f"Scene refresh failed: {e}")
        
        # Update legend if property is set
        if self.session.current_property:
            try:
                self._update_scalar_bar(self.session.current_property)
            except Exception:
                logger.debug("Failed to update scalar bar during refresh", exc_info=True)
        
        self._app.notify_panels_scene_changed()
    
    def fit_to_view(self) -> None:
        """Fit the camera to show all visible objects."""
        if hasattr(self.renderer, "reset_camera") and callable(self.renderer.reset_camera):
            try:
                self.renderer.reset_camera()
            except Exception as e:
                logger.warning(f"Failed to fit camera to view: {e}")
    
    def set_view_preset(self, preset: str) -> None:
        """
        Set camera to a predefined view.
        
        Args:
            preset: View preset name (e.g., "top", "front", "isometric")
        """
        if hasattr(self.renderer, "set_view_preset"):
            try:
                self.renderer.set_view_preset(preset)
            except Exception as e:
                logger.warning(f"Failed to set view preset: {e}")
    
    def set_projection_mode(self, orthographic: bool) -> None:
        """
        Set camera projection mode.
        
        Args:
            orthographic: True for orthographic, False for perspective
        """
        if hasattr(self.renderer, "set_projection_mode"):
            try:
                self.renderer.set_projection_mode(orthographic)
            except Exception as e:
                logger.warning(f"Failed to set projection mode: {e}")
        elif hasattr(self.renderer, "plotter"):
            try:
                if orthographic:
                    self.renderer.plotter.enable_parallel_projection()
                else:
                    self.renderer.plotter.disable_parallel_projection()
            except Exception as e:
                logger.warning(f"Failed to set projection on plotter: {e}")
    
    def set_trackball_mode(self, enabled: bool) -> None:
        """
        Enable or disable trackball interaction mode.
        
        Args:
            enabled: Whether trackball mode should be enabled
        """
        if hasattr(self.renderer, "set_trackball_mode"):
            try:
                self.renderer.set_trackball_mode(enabled)
            except Exception as e:
                logger.warning(f"Failed to set trackball mode: {e}")
        elif hasattr(self.renderer, "plotter"):
            try:
                if enabled:
                    self.renderer.plotter.enable_trackball_style()
                else:
                    self.renderer.plotter.disable_trackball_style()
            except Exception as e:
                logger.warning(f"Failed to set trackball on plotter: {e}")
    
    # =========================================================================
    # Slicing and Filtering
    # =========================================================================
    
    def apply_slice(self, axis: str, position: float) -> None:
        """
        Apply a slice plane to the visualization.
        
        Args:
            axis: Axis for slicing ("x", "y", or "z")
            position: Position along the axis
        """
        self.session.slice_axis = axis
        self.session.slice_position = position
        if hasattr(self.renderer, "apply_slice"):
            try:
                self.renderer.apply_slice(axis, position)
            except Exception as e:
                logger.warning(f"Failed to apply slice: {e}")
    
    def apply_filters(self, filters: Dict[str, tuple]) -> None:
        """
        Apply property filters to the visualization.
        
        Args:
            filters: Dict mapping property names to (min, max) filter ranges
        """
        self.session.filter_specs = filters.copy()
        if hasattr(self.renderer, "apply_filters"):
            try:
                self.renderer.apply_filters(filters)
            except Exception as e:
                logger.warning(f"Failed to apply filters: {e}")
    
    # =========================================================================
    # Render Payloads
    # =========================================================================
    
    def apply_render_payload(self, payload: Any) -> Any:
        """
        Apply a render payload to the renderer (PyVista isolation API).
        
        Args:
            payload: Render payload (MeshPayload, GridPayload, etc.)
            
        Returns:
            Actor(s) or None
        """
        from ..visualization.render_payloads import (
            MeshPayload, GridPayload, CrossSectionPayload, PitShellPayload,
            PointCloudPayload, LinePayload, SecondaryViewPayload
        )
        
        if self.renderer is None:
            logger.warning("Renderer not initialized; cannot apply render payload")
            return None
        
        try:
            if isinstance(payload, MeshPayload):
                return self.renderer.load_mesh(payload)
            elif isinstance(payload, GridPayload):
                return self.renderer.load_grid(payload)
            elif isinstance(payload, CrossSectionPayload):
                return self.renderer.add_cross_section(payload)
            elif isinstance(payload, PitShellPayload):
                return self.renderer.render_pit_shell(payload)
            elif isinstance(payload, PointCloudPayload):
                mesh_payload = MeshPayload(
                    name=payload.name,
                    vertices=payload.points,
                    scalars=payload.scalars,
                    colors=payload.colors,
                    opacity=1.0,
                    visible=payload.visible,
                    metadata=payload.metadata
                )
                return self.renderer.load_mesh(mesh_payload)
            elif isinstance(payload, LinePayload):
                section_payload = CrossSectionPayload(
                    name=payload.name,
                    points=payload.points,
                    lines=payload.lines,
                    thickness=0.0,
                    color=payload.color,
                    opacity=1.0,
                    visible=payload.visible,
                    metadata=payload.metadata
                )
                return self.renderer.add_cross_section(section_payload)
            elif isinstance(payload, SecondaryViewPayload):
                self.renderer.update_secondary_view(payload.view_id, payload)
                return None
            else:
                logger.warning(f"Unknown payload type: {type(payload)}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying render payload: {e}", exc_info=True)
            return None
    
    def apply_results_to_model(self, payload: Dict[str, Any]) -> None:
        """
        Apply finished analysis results to the renderer and legend managers.
        
        Args:
            payload: Result payload produced by analysis task handlers.
        """
        if not payload:
            return

        # Check if we need to create PyVista grid from primitive data (main thread only)
        if payload.get('_create_grid_in_main_thread', False):
            from .grid_builder import create_grid_from_result
            grid = create_grid_from_result(payload)
            if grid:
                payload['grid'] = grid
                # Also set visualization mesh for compatibility
                if 'visualization' not in payload:
                    payload['visualization'] = {}
                payload['visualization']['mesh'] = grid

        visualization = payload.get("visualization") or {}
        mesh = visualization.get("mesh") or payload.get("grid")  # Support both formats
        property_name = visualization.get("property") or payload.get("property_name")
        layer_name = visualization.get("layer_name") or payload.get("name", "Analysis Result")

        if mesh is not None and hasattr(self.renderer, "add_mesh"):
            try:
                self.renderer.add_mesh(
                    mesh,
                    scalars=property_name,
                    name=layer_name,
                    layer_type="analysis",
                    show_edges=False,
                )
                try:
                    self.renderer.set_active_layer_for_controls(layer_name)
                except Exception:
                    logger.debug("Renderer failed to set active layer '%s'", layer_name, exc_info=True)
            except Exception:
                logger.error("Failed to add analysis mesh '%s' to renderer", layer_name, exc_info=True)

        metadata = payload.get("metadata", {})
        if self.legend_manager and property_name:
            legend_payload = {
                "title": layer_name,
                "vmin": metadata.get("estimates_min"),
                "vmax": metadata.get("estimates_max"),
                "colormap": self.session.color_map,
            }
            try:
                self.legend_manager.update_from_property(property_name, legend_payload)
                self.legend_manager.set_visibility(True)
            except Exception:
                logger.debug("Legend update failed for property %s", property_name, exc_info=True)

        self._app.notify_panels_scene_changed()
    
    # =========================================================================
    # Schedule / Pit Visualization
    # =========================================================================
    
    def show_schedule(self, schedule_df, mode: str = "Period") -> None:
        """
        Delegate schedule visualisation to the renderer.
        
        Args:
            schedule_df: Pandas DataFrame describing the schedule.
            mode: Colour mode to apply ("Period", "Phase", "Destination", "Value").
        """
        if hasattr(self.renderer, "show_schedule"):
            try:
                self.renderer.show_schedule(schedule_df, mode)
            except Exception as exc:
                logger.error(f"Renderer failed to display schedule: {exc}", exc_info=True)
        else:
            logger.warning("Renderer does not implement show_schedule().")
    
    def show_optimal_pit(self, pit_df) -> None:
        """
        Delegate optimal pit visualisation to the renderer.
        
        Args:
            pit_df: DataFrame containing optimal pit membership.
        """
        if hasattr(self.renderer, "show_optimal_pit"):
            try:
                self.renderer.show_optimal_pit(pit_df)
            except Exception as exc:
                logger.error(f"Renderer failed to display optimal pit: {exc}", exc_info=True)
        else:
            logger.warning("Renderer does not implement show_optimal_pit().")
    
    def render_pushback_layer(self, plan: Any, style_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Render pushback layer in 3D view.
        
        Args:
            plan: PushbackPlan instance
            style_config: Optional style configuration
        """
        if self.renderer is None:
            logger.warning("Renderer not available")
            return
        
        try:
            block_model = self._app.block_model
            if block_model is None:
                logger.warning("No block model available for pushback rendering")
                return
            
            self.renderer.render_pushbacks(plan, block_model, style_config)
        except Exception as e:
            logger.error(f"Failed to render pushback layer: {e}", exc_info=True)
            raise
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_screenshot(self, path: str, resolution: Tuple[int, int] = (1920, 1080)) -> None:
        """
        Export current view as screenshot.
        
        Args:
            path: File path to save screenshot
            resolution: Tuple of (width, height) in pixels
        """
        if hasattr(self.renderer, 'export_screenshot'):
            self.renderer.export_screenshot(path, resolution)
        elif hasattr(self.renderer, 'plotter'):
            self.renderer.plotter.screenshot(path)
        logger.info(f"Exported screenshot to {path}")
    
    # =========================================================================
    # Export Convenience Methods
    # =========================================================================
    
    def export_resources(self, result: Any, path: Union[str, Path], excel: bool = False) -> None:
        """
        Export resource results to CSV or Excel.
        
        Args:
            result: ResourceResult instance
            path: Output file path
            excel: If True, export to Excel; otherwise CSV
        """
        from ..utils.data_bridge import resource_result_to_dataframe
        from ..utils.export_helpers import export_dataframe_to_csv, export_dataframe_to_excel
        
        df = resource_result_to_dataframe(result)
        if excel:
            export_dataframe_to_excel(df, path, sheet_name="Resource Summary")
        else:
            export_dataframe_to_csv(df, path)
    
    def export_irr_results(self, result: Any, path: Union[str, Path], excel: bool = False) -> None:
        """
        Export IRR results to CSV or Excel.
        
        Args:
            result: IRRResult instance or dict
            path: Output file path
            excel: If True, export to Excel with multiple sheets; otherwise CSV
        """
        from ..utils.data_bridge import irr_result_to_dataframe, schedule_result_to_dataframe
        from ..utils.export_helpers import export_dataframe_to_csv, export_multiple_sheets_to_excel
        import pandas as pd
        
        if excel:
            frames = {}
            frames['Summary'] = irr_result_to_dataframe(result)
            
            if isinstance(result, dict) and 'best_schedule' in result:
                frames['Mining Schedule'] = schedule_result_to_dataframe(result)
            elif hasattr(result, 'best_schedule') and result.best_schedule is not None:
                frames['Mining Schedule'] = schedule_result_to_dataframe(result)
            
            if isinstance(result, dict) and 'npv_distribution' in result:
                frames['NPV Distribution'] = pd.DataFrame({
                    'Scenario': range(len(result['npv_distribution'])),
                    'NPV': result['npv_distribution']
                })
            elif hasattr(result, 'npv_distribution'):
                frames['NPV Distribution'] = pd.DataFrame({
                    'Scenario': range(len(result.npv_distribution)),
                    'NPV': result.npv_distribution
                })
            
            export_multiple_sheets_to_excel(frames, path)
        else:
            df = irr_result_to_dataframe(result)
            export_dataframe_to_csv(df, path)
    
    def export_analysis_results(self, result: Any, path: Union[str, Path], excel: bool = False, result_type: str = "analysis") -> None:
        """
        Export analysis results to CSV or Excel.
        
        Args:
            result: Analysis result (VariogramResult, KrigingResult, SGSIMResult, etc.)
            path: Output file path
            excel: If True, export to Excel; otherwise CSV
            result_type: Type of result ('variogram', 'kriging', 'sgsim', 'swath', 'kmeans', 'uncertainty')
        """
        from ..utils.data_bridge import (
            variogram_result_to_dataframe, kriging_result_to_dataframe,
            sgsim_result_to_dataframe, swath_result_to_dataframe,
            kmeans_result_to_dataframe, uncertainty_result_to_dataframe
        )
        from ..utils.export_helpers import export_dataframe_to_csv, export_dataframe_to_excel, export_multiple_sheets_to_excel
        
        if result_type == 'variogram':
            df = variogram_result_to_dataframe(result)
            if excel:
                export_dataframe_to_excel(df, path, sheet_name="Variogram")
            else:
                export_dataframe_to_csv(df, path)
        elif result_type == 'kriging':
            df = kriging_result_to_dataframe(result)
            if excel:
                export_dataframe_to_excel(df, path, sheet_name="Kriging")
            else:
                export_dataframe_to_csv(df, path)
        elif result_type == 'sgsim':
            df = sgsim_result_to_dataframe(result)
            if excel:
                export_dataframe_to_excel(df, path, sheet_name="SGSIM")
            else:
                export_dataframe_to_csv(df, path)
        elif result_type == 'swath':
            df = swath_result_to_dataframe(result)
            if excel:
                export_dataframe_to_excel(df, path, sheet_name="Swath")
            else:
                export_dataframe_to_csv(df, path)
        elif result_type == 'kmeans':
            df = kmeans_result_to_dataframe(result)
            if excel:
                export_dataframe_to_excel(df, path, sheet_name="KMeans")
            else:
                export_dataframe_to_csv(df, path)
        elif result_type == 'uncertainty':
            frames = uncertainty_result_to_dataframe(result)
            if excel:
                export_multiple_sheets_to_excel(frames, path)
            else:
                if 'summary' in frames:
                    export_dataframe_to_csv(frames['summary'], path)
                else:
                    raise ValueError("Uncertainty result has no summary DataFrame")
        else:
            raise ValueError(f"Unknown result_type: {result_type}")
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _apply_visual_prefs(self) -> None:
        """Apply stored visual preferences to current visualization."""
        if hasattr(self.renderer, 'set_transparency'):
            self.renderer.set_transparency(self.session.transparency)
        
        if self.legend_manager:
            self.legend_manager.set_visibility(self.session.legend_visible)
        elif hasattr(self.renderer, 'toggle_scalar_bar'):
            self.renderer.toggle_scalar_bar(self.session.legend_visible)
        elif hasattr(self.renderer, 'toggle_legend'):
            self.renderer.toggle_legend(self.session.legend_visible)
    
    def _update_scalar_bar(self, title: str) -> None:
        """
        Update legend with current property.
        
        Args:
            title: Title/label for the legend
        """
        if self.legend_manager and self.session.current_property:
            metadata = {
                'vmin': None,
                'vmax': None,
                'colormap': self.session.color_map,
                'title': title
            }
            self.legend_manager.update_from_property(title, metadata)
        elif hasattr(self.renderer, 'update_scalar_bar'):
            self.renderer.update_scalar_bar(title=title, n_labels=5, label_fmt='%.2f')

