"""
File Coordinator — owns all file I/O operations.

Extracted from MainWindow to reduce god-object complexity.
Handles: open, load, recent files, export (data/model/screenshot),
layout composer, block model helpers, clear scene.

Usage:
    # In MainWindow.__init__:
    self._file_coordinator = FileCoordinator(self)

    # Delegates:
    def open_file(self):
        self._file_coordinator.open_file()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt6.QtCore import QObject, QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QMessageBox

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class FileCoordinator(QObject):
    """
    Owns all file I/O operations for MainWindow.

    Methods here were extracted verbatim from MainWindow to preserve
    existing behavior. They access MainWindow via self._mw.
    """

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._mw = main_window

    # ═══════════════════════════════════════════════════════════════
    # OPEN / LOAD
    # ═══════════════════════════════════════════════════════════════

    def open_file(self):
        """Open a block model file via file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._mw,
            "Open Block Model File",
            "",
            "All Supported Files (*.csv *.txt *.vtk *.vtu *.obj *.gltf);;"
            "CSV Files (*.csv);;Text Files (*.txt);;"
            "VTK Files (*.vtk *.vtu);;3D Models (*.obj *.gltf)"
        )
        if not file_path:
            return
        self.load_file(Path(file_path))

    def load_file(self, file_path: Path):
        """
        Load a file in background thread via controller task system.

        SECURITY: Validates file path and size before loading.
        """
        from ..utils.security import (
            FileSizeExceededError,
            SecurityError,
            validate_file_path,
            validate_file_size,
        )

        # Validate path
        try:
            validated_path = validate_file_path(file_path, must_exist=True)
        except SecurityError as e:
            QMessageBox.critical(
                self._mw, "Security Error",
                f"Cannot load file: {e}\n\nPlease select a valid file."
            )
            return
        except FileNotFoundError:
            QMessageBox.critical(
                self._mw, "File Not Found", f"File not found: {file_path}"
            )
            return

        # Check file size
        try:
            file_size = validate_file_size(validated_path, file_type='csv')
            file_size_mb = file_size / (1024 * 1024)
        except FileSizeExceededError as e:
            QMessageBox.critical(
                self._mw, "File Too Large",
                f"{e}\n\nPlease use a smaller file or contact support."
            )
            return
        except Exception as e:
            QMessageBox.critical(self._mw, "Error", f"Cannot read file: {e}")
            return

        # Warn for large files
        if file_size_mb > 50:
            reply = QMessageBox.question(
                self._mw, "Large File Warning",
                f"This file is {file_size_mb:.1f} MB. Loading may take time. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        file_path = validated_path
        self.add_recent_file(file_path)

        mw = self._mw
        if not mw.controller:
            QMessageBox.critical(mw, "Error", "Controller not available for file loading")
            return

        params = {"file_path": file_path}

        def on_load_complete(result: Dict[str, Any]):
            logger.info(f"File load complete. Result keys: {list(result.keys()) if result else 'None'}")
            if result is None or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No result"
                logger.error(f"File load error: {error_msg}")
                self.on_load_error(error_msg)
                return

            block_model = result.get("block_model")
            if block_model is not None:
                result_file_path = result.get("file_path")
                if result_file_path:
                    mw.current_file_path = Path(result_file_path)
                try:
                    self.on_file_loaded(block_model)
                except Exception as e:
                    logger.error(f"Error in on_file_loaded: {type(e).__name__}: {e}")
                    self.on_load_error(f"Failed to process loaded file: {e}")
            else:
                logger.error("No block model in result")
                self.on_load_error("No block model in result")

        mw.controller.run_task('load_file', params, callback=on_load_complete)
        logger.info(f"Started loading file via task system: {file_path}")

    def on_file_loaded(self, block_model):
        """Handle successful file load."""
        from ..models.block_model import BlockModel

        mw = self._mw
        handler_start = time.time()

        try:
            # Register with coordinate manager
            from ...utils.coordinate_manager import CoordinateBounds, AlignmentStatus
            file_name = mw.current_file_path.name if mw.current_file_path else "Unknown"
            dataset_name = f"Block Model: {file_name}"
            bounds = block_model.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            cb = CoordinateBounds(
                xmin=bounds[0], xmax=bounds[1],
                ymin=bounds[2], ymax=bounds[3],
                zmin=bounds[4], zmax=bounds[5],
            )
            dataset_info = mw.coordinate_manager.register_dataset(
                dataset_name, cb, block_model.block_count, 'block_model'
            )

            # Coordinate alignment dialog
            if dataset_info.alignment_status == AlignmentStatus.POTENTIALLY_MISALIGNED:
                logger.warning("Block model may be in a different coordinate system!")
                reply = QMessageBox.question(
                    mw, "Coordinate Alignment Warning",
                    f"The block model appears to be in a different coordinate system.\n\n"
                    f"Block model center: ({dataset_info.bounds.center[0]:,.2f}, "
                    f"{dataset_info.bounds.center[1]:,.2f}, {dataset_info.bounds.center[2]:,.2f})\n\n"
                    f"Continue loading anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    mw.coordinate_manager.confirm_alignment(
                        dataset_name, "User confirmed at load time"
                    )

            mw.current_model = block_model

            # Register with DataRegistry — triggers _on_block_model_loaded_from_registry
            try:
                registry = mw.controller.registry if mw.controller else None
                if registry is not None:
                    registry.register_block_model(
                        block_model,
                        source_panel="MainWindow",
                        metadata={
                            "source_path": str(mw.current_file_path)
                            if mw.current_file_path else "Unknown"
                        }
                    )
            except Exception as exc:
                logger.debug(f"Failed to register in DataRegistry: {exc}", exc_info=True)
                # Fallback: update viewer directly
                if mw.viewer_widget:
                    mw.viewer_widget.refresh_scene(block_model)
                if mw.property_panel:
                    mw.property_panel.set_block_model(block_model)

            handler_time = time.time() - handler_start
            logger.info(f"PERF: on_file_loaded total: {handler_time:.3f}s")

        except Exception as e:
            logger.error(f"Error in on_file_loaded: {type(e).__name__}: {e}")
            QMessageBox.critical(mw, "File Load Error",
                                 f"Error processing loaded file:\n{e}")

    def on_block_model_loaded_from_registry(self, block_model):
        """
        Handle block model loaded signal from DataRegistry.
        Single source of truth for viewer and panel updates.
        """
        mw = self._mw

        # Skip expensive viewer refreshes when bulk-restoring registry models
        # during project load.  A final refresh happens after restore completes.
        if getattr(mw, '_restoring_registry_models', False):
            logger.debug("Skipping viewer refresh during registry model restoration")
            return

        handler_start = time.time()

        try:
            # Update viewer
            if mw.viewer_widget:
                mw.viewer_widget.refresh_scene(block_model)

            # Force app state update
            if mw.controller:
                try:
                    mw.controller._update_state_from_scene()
                except Exception:
                    pass

            # Update property panel
            if mw.property_panel:
                mw.property_panel.set_block_model(block_model)

            handler_time = time.time() - handler_start
            logger.info(f"PERF: _on_block_model_loaded total: {handler_time:.3f}s")

            # Persist last_file for session restore
            try:
                settings = QSettings("GeoX", "Session")
                if mw.current_file_path:
                    settings.setValue("last_file", str(mw.current_file_path))
            except Exception:
                pass

            # Restore drillhole data
            try:
                if getattr(mw, '_pending_drillhole_state', None):
                    mw._restore_drillhole_data(mw._pending_drillhole_state)
                    mw._pending_drillhole_state = None
            except Exception as e:
                logger.warning(f"Failed to restore drillhole data: {e}")

            # Restore registry models (variogram, kriging, SGSIM, etc.)
            # Clear before restoring to prevent re-entrant calls.
            try:
                _pending = getattr(mw, '_pending_registry_models_state', None)
                if _pending:
                    mw._pending_registry_models_state = None
                    mw._restoring_registry_models = True
                    try:
                        mw._restore_registry_models(_pending)
                    finally:
                        mw._restoring_registry_models = False
            except Exception as e:
                logger.warning(f"Failed to restore registry models: {e}")

            # After ALL data is restored, refresh the viewer with the correct
            # current model from the registry (which has all properties).
            try:
                registry = mw.controller.registry if mw.controller else None
                if registry is not None:
                    current_model = registry.get_block_model(copy_data=False)
                    if current_model is not None and mw.viewer_widget:
                        mw.viewer_widget.refresh_scene(current_model)
                        if mw.property_panel:
                            mw.property_panel.set_block_model(current_model)
                        logger.info("Final viewer refresh with current registry model after project restore")
            except Exception as e:
                logger.warning(f"Failed final viewer refresh after restore: {e}")

            # Apply renderer state AFTER all data restored so saved properties
            # (kr_Cu, PIT_SHELL, etc.) exist on the block model.
            try:
                if getattr(mw, '_pending_session_state', None) and mw.viewer_widget and mw.viewer_widget.renderer:
                    mw.viewer_widget.renderer.apply_session_state(mw._pending_session_state)
                    mw._pending_session_state = None
            except Exception as e:
                logger.warning(f"Failed to apply pending session state: {e}")

            # Enable model-dependent actions
            try:
                if hasattr(mw, 'view_data_action') and mw.view_data_action:
                    mw.view_data_action.setEnabled(True)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error loading model in viewer: {e}")
            QMessageBox.critical(mw, "Error", f"Failed to load model: {e}")

    def on_load_error(self, error_message: str):
        """Handle file load error."""
        QMessageBox.critical(
            self._mw, "Load Error",
            f"Failed to load file:\n\n{error_message}"
        )
        logger.error(f"File load error: {error_message}")

    # ═══════════════════════════════════════════════════════════════
    # RECENT FILES
    # ═══════════════════════════════════════════════════════════════

    def add_recent_file(self, file_path: Path):
        """Add a file to the recent files list."""
        mw = self._mw
        try:
            file_str = str(file_path.absolute())
            recent_files = mw.config.config.get('ui', {}).get('recent_files', [])

            if file_str in recent_files:
                recent_files.remove(file_str)
            recent_files.insert(0, file_str)

            max_recent = mw.config.config.get('ui', {}).get('max_recent_files', 10)
            recent_files = recent_files[:max_recent]

            if 'ui' not in mw.config.config:
                mw.config.config['ui'] = {}
            mw.config.config['ui']['recent_files'] = recent_files
            mw.config.save_config()

            self.update_recent_files_menu()
        except Exception as e:
            logger.warning(f"Failed to add recent file: {e}")

    def update_recent_files_menu(self):
        """Update the recent files submenu."""
        mw = self._mw
        try:
            mw.recent_files_menu.clear()
            recent_files = mw.config.config.get('ui', {}).get('recent_files', [])

            if not recent_files:
                no_files_action = QAction("(No recent files)", mw)
                no_files_action.setEnabled(False)
                mw.recent_files_menu.addAction(no_files_action)
                return

            for i, file_path_str in enumerate(recent_files):
                file_path = Path(file_path_str)
                action_text = f"{i+1}. {file_path.name}"
                action = QAction(action_text, mw)
                action.setStatusTip(file_path_str)
                action.setToolTip(file_path_str)
                if i < 9:
                    action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
                action.triggered.connect(
                    lambda checked, fp=file_path: self.open_recent_file(fp)
                )
                mw.recent_files_menu.addAction(action)

            mw.recent_files_menu.addSeparator()
            clear_action = QAction("Clear Recent Files", mw)
            clear_action.triggered.connect(self.clear_recent_files)
            mw.recent_files_menu.addAction(clear_action)
        except Exception as e:
            logger.warning(f"Failed to update recent files menu: {e}")

    def open_recent_file(self, file_path: Path):
        """Open a file from the recent files list."""
        if not file_path.exists():
            QMessageBox.warning(
                self._mw, "File Not Found",
                f"The file no longer exists:\n\n{file_path}"
            )
            try:
                mw = self._mw
                recent_files = mw.config.config.get('ui', {}).get('recent_files', [])
                file_str = str(file_path.absolute())
                if file_str in recent_files:
                    recent_files.remove(file_str)
                    mw.config.config['ui']['recent_files'] = recent_files
                    mw.config.save_config()
                    self.update_recent_files_menu()
            except Exception:
                pass
            return
        self.load_file(file_path)

    def clear_recent_files(self):
        """Clear the recent files list."""
        mw = self._mw
        reply = QMessageBox.question(
            mw, "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if 'ui' not in mw.config.config:
                mw.config.config['ui'] = {}
            mw.config.config['ui']['recent_files'] = []
            mw.config.save_config()
            self.update_recent_files_menu()
            mw.status_bar.showMessage("Recent files cleared", 2000)

    # ═══════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════

    def export_screenshot(self, filename: str = ""):
        """Export current view as screenshot."""
        mw = self._mw
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(
                mw, "Save Screenshot", "screenshot.png",
                "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
        if filename and mw.viewer_widget:
            mw.viewer_widget.export_screenshot(filename)
            mw.status_bar.showMessage(f"Screenshot saved: {filename}", 3000)
            logger.info(f"Screenshot saved: {filename}")

    def export_filtered_data(self):
        """Open comprehensive data export dialog."""
        from ..screenshot_export_dialog import ScreenshotExportDialog
        from ..data_export_dialog import DataExportDialog

        mw = self._mw
        try:
            dialog = DataExportDialog(mw.registry, mw)
            dialog.exec()
        except Exception as e:
            logger.error(f"Failed to open export dialog: {e}", exc_info=True)
            QMessageBox.critical(mw, "Export Error",
                                 f"Failed to open export dialog:\n{e}")

    def export_model(self):
        """Export 3D model to file."""
        mw = self._mw
        if not self.has_valid_block_model():
            QMessageBox.warning(mw, "No Model", "Load a model first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            mw, "Export 3D Model", "block_model.stl",
            "STL Files (*.stl);;OBJ Files (*.obj);;VTK Files (*.vtk)"
        )
        if filename and mw.viewer_widget:
            try:
                mw.viewer_widget.export_mesh_to_file(filename)
                mw.status_bar.showMessage(f"Model exported: {filename}", 3000)
                logger.info(f"Exported 3D model: {filename}")
            except Exception as e:
                QMessageBox.critical(mw, "Export Error",
                                     f"Failed to export model:\n{e}")
                logger.error(f"Export error: {e}")

    def open_layout_composer(self):
        """Open the Layout Composer window."""
        from ..layout.layout_window import LayoutComposerWindow

        mw = self._mw
        if not hasattr(mw, '_layout_composer') or mw._layout_composer is None:
            mw._layout_composer = LayoutComposerWindow(main_window=mw, parent=mw)
        mw._layout_composer.show()
        mw._layout_composer.raise_()
        mw._layout_composer.activateWindow()
        logger.info("Opened Layout Composer")

    def quick_layout_export(self, export_type: str = "pdf"):
        """Quick export with a standard layout template."""
        from ..layout.layout_document import (
            LayoutDocument, ViewportItem, LegendItem, ScaleBarItem, TextItem,
        )
        from ..layout.layout_export import export_pdf, export_png

        mw = self._mw

        if export_type == "png_hd":
            dpi, ext, fmt = 600, ".png", "png"
        elif export_type == "png":
            dpi, ext, fmt = 300, ".png", "png"
        else:
            dpi, ext, fmt = 300, ".pdf", "pdf"

        default_name = f"GeoX_Export_{dpi}dpi{ext}"
        filepath, _ = QFileDialog.getSaveFileName(
            mw, f"Quick Export to {fmt.upper()}",
            str(Path.home() / default_name),
            f"{fmt.upper()} Files (*{ext})"
        )
        if not filepath:
            return
        filepath = Path(filepath)

        doc = LayoutDocument(name="Quick Export")
        viewport = ViewportItem(name="Main View", x_mm=10, y_mm=25,
                                width_mm=200, height_mm=150)
        if mw.viewer_widget and hasattr(mw.viewer_widget, 'renderer'):
            renderer = mw.viewer_widget.renderer
            if hasattr(renderer, 'get_camera_info'):
                viewport.camera_state = renderer.get_camera_info()
            if hasattr(renderer, 'legend_manager') and hasattr(renderer.legend_manager, 'get_state'):
                viewport.legend_state = renderer.legend_manager.get_state()
        doc.add_item(viewport)
        doc.add_item(LegendItem(name="Legend", x_mm=220, y_mm=25,
                                width_mm=60, height_mm=100,
                                legend_state=viewport.legend_state))
        doc.add_item(ScaleBarItem(name="Scale Bar", x_mm=10, y_mm=185,
                                  width_mm=60, height_mm=12))
        doc.add_item(TextItem(name="Title", text="GeoX Export",
                              x_mm=10, y_mm=5, width_mm=277, height_mm=15,
                              font_size=16, font_bold=True, alignment="center"))

        import getpass
        from datetime import datetime
        metadata_values = {
            "project_name": "GeoX Project",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": getpass.getuser(),
            "software_version": "GeoX",
            "export_dpi": str(dpi),
        }

        try:
            if fmt == "pdf":
                export_pdf(doc, filepath, dpi, mw.viewer_widget, metadata_values)
            else:
                export_png(doc, filepath, dpi, mw.viewer_widget, metadata_values)

            mw.status_bar.showMessage(f"Exported to {filepath}", 5000)
            logger.info(f"Quick layout exported to {filepath}")
            QMessageBox.information(
                mw, "Export Complete",
                f"Layout exported to:\n{filepath}\n\nAudit record saved alongside."
            )
        except Exception as e:
            logger.error(f"Quick export failed: {e}")
            QMessageBox.warning(mw, "Export Failed", f"Failed to export: {e}")

    # ═══════════════════════════════════════════════════════════════
    # CLEAR SCENE
    # ═══════════════════════════════════════════════════════════════

    def clear_scene(self):
        """Clear the scene and remove all models."""
        mw = self._mw
        has_block_model = self.has_valid_block_model()
        has_drillhole_data = False
        has_any_layers = False

        # Check renderer active_layers for drillholes and other content
        renderer = getattr(mw.viewer_widget, 'renderer', None) if mw.viewer_widget else None
        if renderer and hasattr(renderer, 'active_layers') and renderer.active_layers:
            has_any_layers = True
            has_drillhole_data = 'drillholes' in renderer.active_layers

        if not has_block_model and not has_drillhole_data and not has_any_layers:
            QMessageBox.information(mw, "No Model", "Scene is already empty.")
            return

        model_types = []
        if has_block_model:
            model_types.append("block model")
        if has_drillhole_data:
            model_types.append("drillholes")
        if has_any_layers and not model_types:
            model_types.append("all layers")
        model_text = " and ".join(model_types)

        reply = QMessageBox.question(
            mw, "Clear Scene",
            f"Are you sure you want to remove the {model_text} and clear the scene?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if mw.viewer_widget:
                mw.viewer_widget.clear_scene()

            # clear_all_layers is also called inside viewer_widget.clear_scene(),
            # but call again in case viewer_widget is None
            if renderer and hasattr(renderer, 'clear_all_layers'):
                renderer.clear_all_layers()

            # ── SIA-001: Clear DataRegistry so old data doesn't bleed through ──
            try:
                from ...core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    DataRegistry._instance.clear_all()
                    logger.info("DataRegistry cleared on scene clear")
            except Exception as e:
                logger.debug(f"Could not clear DataRegistry on scene clear: {e}")

            # ── SIA-001: Clear ALL instantiated panels, not just 3 hardcoded ones ──
            if hasattr(mw, 'panel_manager') and mw.panel_manager is not None:
                try:
                    mw.panel_manager.clear_all_panels()
                except Exception as e:
                    logger.debug(f"PanelManager clear_all_panels failed: {e}")

            # Legacy fallback for panels not registered with PanelManager
            if mw.property_panel:
                mw.property_panel.clear()
            if mw.data_viewer_panel and mw.data_viewer_panel.isVisible():
                mw.data_viewer_panel.clear()
            if hasattr(mw, 'scene_inspector_panel') and mw.scene_inspector_panel:
                mw.scene_inspector_panel.clear_camera_info()

            mw.current_model = None
            mw.current_file_path = None

            if hasattr(mw, 'view_data_action') and mw.view_data_action:
                try:
                    mw.view_data_action.setEnabled(False)
                except Exception:
                    pass

            # Clear undo history
            if mw.controller and hasattr(mw.controller, 'undo_manager'):
                mw.controller.undo_manager.clear()

            mw.status_bar.showMessage("Scene cleared", 3000)
            mw.setWindowTitle("GeoX")
            logger.info("Scene cleared successfully")

        except Exception as e:
            QMessageBox.critical(mw, "Clear Error", f"Failed to clear scene:\n{e}")
            logger.error(f"Clear scene error: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════════════════
    # BLOCK MODEL HELPERS
    # ═══════════════════════════════════════════════════════════════

    def has_valid_block_model(self) -> bool:
        """Check if a block model exists (current, renderer layers, or registry)."""
        import pandas as pd

        mw = self._mw

        if mw.current_model is not None:
            if isinstance(mw.current_model, pd.DataFrame):
                if not mw.current_model.empty:
                    return True
            else:
                return True

        if self.has_block_model_in_layers():
            return True

        try:
            registry = mw.controller.registry if mw.controller else None
            if registry:
                classified = registry.get_classified_block_model(copy_data=False)
                if classified is not None:
                    if isinstance(classified, pd.DataFrame):
                        return not classified.empty
                    return True
                block_model = registry.get_block_model(copy_data=False)
                if block_model is not None:
                    if isinstance(block_model, pd.DataFrame):
                        return not block_model.empty
                    return True
        except Exception as e:
            logger.debug(f"Error checking registry for block model: {e}")

        return False

    def has_block_model_in_layers(self) -> bool:
        """Check if there's block model data in renderer layers."""
        mw = self._mw
        try:
            if not (mw.viewer_widget and hasattr(mw.viewer_widget, 'renderer')):
                return False
            renderer = mw.viewer_widget.renderer
            if hasattr(renderer, 'active_layers') and renderer.active_layers:
                block_types = (
                    'blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                    'classification', 'resource', 'estimate'
                )
                patterns = [
                    'sgsim', 'kriging', 'simulation', 'block',
                    'classification', 'resource', 'estimate',
                    'measured', 'indicated', 'inferred',
                    'ordinary', 'simple', 'universal', 'indicator',
                    'cosgsim', 'sis', 'turning', 'dbs', 'mps', 'grf'
                ]
                for layer_name, layer_info in renderer.active_layers.items():
                    if layer_info.get('type', '') in block_types:
                        if layer_info.get('data') is not None:
                            return True
                    layer_lower = layer_name.lower()
                    if any(p in layer_lower for p in patterns):
                        if layer_info.get('data') is not None:
                            return True
            if hasattr(renderer, 'block_meshes') and renderer.block_meshes:
                if 'unstructured_grid' in renderer.block_meshes:
                    return True
        except Exception as e:
            logger.debug(f"Error checking block model in layers: {e}")
        return False

    def get_block_model_from_layers(self):
        """Get block model DataFrame from renderer layers or registry."""
        import pandas as pd

        mw = self._mw

        # Try renderer layers
        try:
            if mw.viewer_widget and hasattr(mw.viewer_widget, 'renderer'):
                renderer = mw.viewer_widget.renderer
                if hasattr(renderer, 'active_layers') and renderer.active_layers:
                    priority = [
                        'sgsim:', 'cosgsim:', 'sis:', 'turning:', 'dbs:',
                        'mps:', 'grf:', 'kriging', 'ordinary:', 'simple:',
                        'universal:', 'indicator:', 'classification',
                        'measured', 'indicated', 'inferred',
                        'resource', 'reserve', 'block'
                    ]
                    for pattern in priority:
                        for lname, linfo in renderer.active_layers.items():
                            if pattern in lname.lower():
                                grid = linfo.get('data')
                                if grid is not None:
                                    df = self._grid_to_dataframe(grid, lname)
                                    if df is not None:
                                        return df

                    # Fallback: any non-drillhole block-type layer
                    block_types = (
                        'blocks', 'volume', 'sgsim', 'kriging', 'simulation',
                        'classification', 'resource', 'estimate'
                    )
                    for lname, linfo in renderer.active_layers.items():
                        if 'drillhole' not in lname.lower():
                            if linfo.get('type', '') in block_types:
                                grid = linfo.get('data')
                                if grid is not None:
                                    df = self._grid_to_dataframe(grid, lname)
                                    if df is not None:
                                        return df
        except Exception as e:
            logger.debug(f"Error getting block model from layers: {e}")

        # Registry fallback
        try:
            registry = mw.controller.registry if mw.controller else None
            if registry:
                for getter in [registry.get_classified_block_model,
                               registry.get_block_model]:
                    obj = getter(copy_data=True)
                    if obj is not None:
                        if isinstance(obj, pd.DataFrame) and not obj.empty:
                            return obj
                        if hasattr(obj, 'to_dataframe'):
                            df = obj.to_dataframe()
                            if df is not None and not df.empty:
                                return df
        except Exception as e:
            logger.debug(f"Error getting block model from registry: {e}")

        return None

    @staticmethod
    def _grid_to_dataframe(grid_data, layer_name):
        """Convert PyVista grid to DataFrame."""
        import numpy as np
        import pandas as pd

        try:
            if isinstance(grid_data, dict) and 'mesh' in grid_data:
                grid_data = grid_data['mesh']

            if hasattr(grid_data, 'cell_centers') and hasattr(grid_data, 'cell_data'):
                centers = grid_data.cell_centers()
                df_data = {
                    'X': centers.points[:, 0],
                    'Y': centers.points[:, 1],
                    'Z': centers.points[:, 2],
                }

                # FIX: Preserve grid spacing as explicit DX/DY/DZ columns.
                # Without these, BlockModel._infer_dimensions_from_positions()
                # uses np.unique() on float64 coordinates and collapses block
                # dimensions to ~1e-13 under float jitter.
                if hasattr(grid_data, 'spacing') and grid_data.spacing is not None:
                    spacing = grid_data.spacing
                    n_cells = len(centers.points)
                    df_data['DX'] = np.full(n_cells, float(spacing[0]), dtype=np.float64)
                    df_data['DY'] = np.full(n_cells, float(spacing[1]), dtype=np.float64)
                    df_data['DZ'] = np.full(n_cells, float(spacing[2]), dtype=np.float64)

                for prop_name in grid_data.cell_data.keys():
                    try:
                        prop_data = grid_data.cell_data[prop_name]
                        if np.issubdtype(prop_data.dtype, np.number):
                            df_data[prop_name] = prop_data
                    except Exception:
                        pass
                if df_data:
                    return pd.DataFrame(df_data)
        except Exception as e:
            logger.debug(f"Error converting grid to DataFrame: {e}")
        return None
