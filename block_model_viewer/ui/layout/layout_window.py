"""
Layout Composer Window for GeoX.

Main window for creating and editing print layouts with viewports,
legends, scale bars, text, and other elements for export to PDF/PNG/TIFF.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from ..modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QStatusBar, QWidget,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QMenu,
    QComboBox, QSpinBox, QLabel, QCheckBox, QDialog, QDialogButtonBox,
    QFormLayout, QGroupBox
)

from ...layout.layout_document import (
    LayoutDocument, PageSpec, PageSize, PageOrientation,
    ViewportItem, LegendItem, ScaleBarItem, NorthArrowItem,
    TextItem, ImageItem, MetadataItem
)
from ...layout.layout_io import (
    save_layout, load_layout, get_layout_file_filter,
    suggest_layout_filename, LAYOUT_FILE_EXTENSION
)
from ...layout.layout_export import (
    export_pdf, export_png, export_tiff, get_export_options, DPI_PRESETS
)
from .layout_canvas import LayoutCanvas
from .layout_item_list import LayoutItemList
from .layout_property_panel import LayoutPropertyPanel

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


class LayoutComposerWindow(QMainWindow):
    """
    Main Layout Composer window.

    Provides a visual editor for creating print layouts with:
    - Central canvas for page editing
    - Left panel for item list management
    - Right panel for property editing
    - Toolbar for adding items
    - Export to PDF, PNG, TIFF with audit records
    """

    # Signals
    layout_saved = pyqtSignal(str)  # filepath
    layout_exported = pyqtSignal(str, str)  # filepath, format

    def __init__(self, main_window: Optional["MainWindow"] = None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._document = LayoutDocument()
        self._current_filepath: Optional[Path] = None
        self._modified = False

        self.setWindowTitle("Layout Composer - GeoX")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Apply dark theme to match GeoX
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d30;
            }
            QDockWidget {
                background-color: #2d2d30;
                color: #f0f0f0;
            }
            QDockWidget::title {
                background-color: #3c3c3c;
                padding: 6px;
            }
            QToolBar {
                background-color: #3c3c3c;
                border: none;
                spacing: 4px;
                padding: 4px;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px;
                color: #f0f0f0;
            }
            QToolButton:hover {
                background-color: #505050;
                border-color: #606060;
            }
            QToolButton:pressed {
                background-color: #404040;
            }
            QStatusBar {
                background-color: #007acc;
                color: white;
            }
            QMenu {
                background-color: #2d2d30;
                color: #f0f0f0;
                border: 1px solid #3c3c3c;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
        """)

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._connect_signals()

        # Create default layout
        self._document.create_default_layout()
        self._refresh_canvas()

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()

    @property
    def document(self) -> LayoutDocument:
        """Get the current document."""
        return self._document

    def _setup_ui(self) -> None:
        """Initialize UI components."""
        # Central canvas
        self._canvas = LayoutCanvas(self._document)
        self.setCentralWidget(self._canvas)

        # Left dock: Item list
        self._item_list_dock = QDockWidget("Items", self)
        self._item_list_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._item_list = LayoutItemList(self._document)
        self._item_list_dock.setWidget(self._item_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._item_list_dock)

        # Right dock: Property panel
        self._property_dock = QDockWidget("Properties", self)
        self._property_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._property_panel = LayoutPropertyPanel()
        self._property_dock.setWidget(self._property_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._property_dock)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _setup_menus(self) -> None:
        """Build menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #3c3c3c;
                color: #f0f0f0;
            }
            QMenuBar::item:selected {
                background-color: #094771;
            }
        """)

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Layout", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_layout)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Layout...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_layout)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_layout)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_layout_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        close_action = QAction("&Close", self)
        close_action.setShortcut(QKeySequence.StandardKey.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.setEnabled(False)  # TODO: Implement undo
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.setEnabled(False)  # TODO: Implement redo
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        delete_action = QAction("&Delete Selected", self)
        delete_action.setShortcut(QKeySequence.StandardKey.Delete)
        delete_action.triggered.connect(self._delete_selected)
        edit_menu.addAction(delete_action)

        select_all_action = QAction("Select &All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self._select_all)
        edit_menu.addAction(select_all_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_fit_action = QAction("&Fit Page", self)
        zoom_fit_action.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit_action.triggered.connect(self._canvas.fit_page_in_view)
        view_menu.addAction(zoom_fit_action)

        view_menu.addSeparator()

        self._grid_action = QAction("Show &Grid", self)
        self._grid_action.setCheckable(True)
        self._grid_action.setChecked(True)
        self._grid_action.triggered.connect(self._toggle_grid)
        view_menu.addAction(self._grid_action)

        self._snap_action = QAction("&Snap to Grid", self)
        self._snap_action.setCheckable(True)
        self._snap_action.setChecked(True)
        self._snap_action.triggered.connect(self._toggle_snap)
        view_menu.addAction(self._snap_action)

        # Insert menu
        insert_menu = menubar.addMenu("&Insert")

        viewport_action = QAction("&Viewport", self)
        viewport_action.triggered.connect(lambda: self._add_item("viewport"))
        insert_menu.addAction(viewport_action)

        legend_action = QAction("&Legend", self)
        legend_action.triggered.connect(lambda: self._add_item("legend"))
        insert_menu.addAction(legend_action)

        scale_bar_action = QAction("&Scale Bar", self)
        scale_bar_action.triggered.connect(lambda: self._add_item("scale_bar"))
        insert_menu.addAction(scale_bar_action)

        north_arrow_action = QAction("&North Arrow", self)
        north_arrow_action.triggered.connect(lambda: self._add_item("north_arrow"))
        insert_menu.addAction(north_arrow_action)

        insert_menu.addSeparator()

        text_action = QAction("&Text", self)
        text_action.triggered.connect(lambda: self._add_item("text"))
        insert_menu.addAction(text_action)

        image_action = QAction("&Image/Logo...", self)
        image_action.triggered.connect(self._add_image)
        insert_menu.addAction(image_action)

        metadata_action = QAction("&Metadata Block", self)
        metadata_action.triggered.connect(lambda: self._add_item("metadata"))
        insert_menu.addAction(metadata_action)

        # Layout menu
        layout_menu = menubar.addMenu("&Layout")

        page_setup_action = QAction("&Page Setup...", self)
        page_setup_action.triggered.connect(self._show_page_setup)
        layout_menu.addAction(page_setup_action)

        layout_menu.addSeparator()

        default_layout_action = QAction("Apply &Default Layout", self)
        default_layout_action.triggered.connect(self._apply_default_layout)
        layout_menu.addAction(default_layout_action)

        # Export menu
        export_menu = menubar.addMenu("E&xport")

        pdf_action = QAction("Export to &PDF...", self)
        pdf_action.triggered.connect(lambda: self._export("pdf"))
        export_menu.addAction(pdf_action)

        png_action = QAction("Export to P&NG...", self)
        png_action.triggered.connect(lambda: self._export("png"))
        export_menu.addAction(png_action)

        tiff_action = QAction("Export to &TIFF...", self)
        tiff_action.triggered.connect(lambda: self._export("tiff"))
        export_menu.addAction(tiff_action)

    def _setup_toolbar(self) -> None:
        """Setup the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add item buttons
        toolbar.addAction("Viewport", lambda: self._add_item("viewport"))
        toolbar.addAction("Legend", lambda: self._add_item("legend"))
        toolbar.addAction("Scale Bar", lambda: self._add_item("scale_bar"))
        toolbar.addAction("North Arrow", lambda: self._add_item("north_arrow"))
        toolbar.addSeparator()
        toolbar.addAction("Text", lambda: self._add_item("text"))
        toolbar.addAction("Image", self._add_image)
        toolbar.addAction("Metadata", lambda: self._add_item("metadata"))
        toolbar.addSeparator()

        # Capture viewport button
        capture_action = toolbar.addAction("Capture Viewport")
        capture_action.triggered.connect(self._capture_current_viewport)

        toolbar.addSeparator()

        # Zoom controls
        toolbar.addAction("Fit", self._canvas.fit_page_in_view)

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Canvas signals
        self._canvas.selection_changed.connect(self._on_selection_changed)
        self._canvas.item_double_clicked.connect(self._on_item_double_clicked)
        self._canvas.canvas_context_menu.connect(self._show_canvas_context_menu)

        # Item list signals
        self._item_list.item_selected.connect(self._on_list_item_selected)
        self._item_list.visibility_changed.connect(self._on_visibility_changed)
        self._item_list.item_deleted.connect(self._on_item_deleted)

        # Property panel signals
        self._property_panel.property_changed.connect(self._on_property_changed)

    def _refresh_canvas(self) -> None:
        """Refresh the canvas with current document items."""
        self._canvas.set_document(self._document)
        self._canvas.refresh_all_items()
        self._item_list.refresh()

    def _mark_modified(self) -> None:
        """Mark document as modified."""
        self._modified = True
        title = f"Layout Composer - {self._document.name}"
        if self._modified:
            title += " *"
        self.setWindowTitle(title)

    # =========================================================================
    # File Operations
    # =========================================================================

    def _new_layout(self) -> None:
        """Create a new layout."""
        if self._modified:
            result = QMessageBox.question(
                self, "Unsaved Changes",
                "Do you want to save changes before creating a new layout?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if result == QMessageBox.StandardButton.Save:
                self._save_layout()
            elif result == QMessageBox.StandardButton.Cancel:
                return

        self._document = LayoutDocument()
        self._document.create_default_layout()
        self._current_filepath = None
        self._modified = False
        self._refresh_canvas()
        self.setWindowTitle("Layout Composer - Untitled")
        self._status_bar.showMessage("New layout created")

    def _open_layout(self) -> None:
        """Open an existing layout file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Layout",
            str(Path.home()),
            get_layout_file_filter()
        )
        if not filepath:
            return

        document = load_layout(Path(filepath))
        if document:
            self._document = document
            self._current_filepath = Path(filepath)
            self._modified = False
            self._refresh_canvas()
            self.setWindowTitle(f"Layout Composer - {self._document.name}")
            self._status_bar.showMessage(f"Loaded {filepath}")
        else:
            QMessageBox.warning(self, "Error", f"Failed to load layout: {filepath}")

    def _save_layout(self) -> None:
        """Save the current layout."""
        if self._current_filepath:
            if save_layout(self._document, self._current_filepath):
                self._modified = False
                self.setWindowTitle(f"Layout Composer - {self._document.name}")
                self._status_bar.showMessage(f"Saved to {self._current_filepath}")
                self.layout_saved.emit(str(self._current_filepath))
            else:
                QMessageBox.warning(self, "Error", "Failed to save layout")
        else:
            self._save_layout_as()

    def _save_layout_as(self) -> None:
        """Save layout to a new file."""
        default_name = suggest_layout_filename(self._document.name.replace(" ", "_"))
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Layout As",
            str(Path.home() / default_name),
            get_layout_file_filter()
        )
        if not filepath:
            return

        self._current_filepath = Path(filepath)
        self._save_layout()

    # =========================================================================
    # Item Operations
    # =========================================================================

    def _add_item(self, item_type: str) -> None:
        """Add a new item to the layout."""
        page = self._document.page
        center_x = page.margin_left_mm + page.content_width_mm / 2 - 25
        center_y = page.margin_top_mm + page.content_height_mm / 2 - 25

        item_factories = {
            "viewport": lambda: ViewportItem(
                name="Viewport",
                x_mm=center_x - 50, y_mm=center_y - 40,
                width_mm=150, height_mm=100,
            ),
            "legend": lambda: LegendItem(
                name="Legend",
                x_mm=center_x + 60, y_mm=center_y - 40,
                width_mm=40, height_mm=80,
            ),
            "scale_bar": lambda: ScaleBarItem(
                name="Scale Bar",
                x_mm=center_x - 30, y_mm=center_y + 50,
                width_mm=60, height_mm=12,
            ),
            "north_arrow": lambda: NorthArrowItem(
                name="North Arrow",
                x_mm=center_x + 70, y_mm=center_y + 50,
                width_mm=20, height_mm=25,
            ),
            "text": lambda: TextItem(
                name="Text",
                text="Enter text here",
                x_mm=center_x - 30, y_mm=center_y - 60,
                width_mm=60, height_mm=15,
            ),
            "metadata": lambda: MetadataItem(
                name="Metadata",
                x_mm=center_x - 35, y_mm=center_y + 70,
                width_mm=70, height_mm=35,
            ),
        }

        factory = item_factories.get(item_type)
        if factory:
            item = factory()
            self._document.add_item(item)
            self._canvas.add_item_graphic(item)
            self._canvas.select_item(item.id)
            self._item_list.refresh()
            self._mark_modified()
            self._status_bar.showMessage(f"Added {item_type}")

    def _add_image(self) -> None:
        """Add an image/logo item."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.svg *.bmp);;All Files (*)"
        )
        if not filepath:
            return

        page = self._document.page
        center_x = page.margin_left_mm + page.content_width_mm / 2
        center_y = page.margin_top_mm + page.content_height_mm / 2

        item = ImageItem(
            name=Path(filepath).stem,
            image_path=filepath,
            x_mm=center_x - 15, y_mm=center_y - 15,
            width_mm=30, height_mm=30,
        )
        self._document.add_item(item)
        self._canvas.add_item_graphic(item)
        self._canvas.select_item(item.id)
        self._item_list.refresh()
        self._mark_modified()
        self._status_bar.showMessage(f"Added image: {Path(filepath).name}")

    def _delete_selected(self) -> None:
        """Delete selected items."""
        selected_ids = self._canvas.get_selected_item_ids()
        for item_id in selected_ids:
            self._document.remove_item(item_id)
            self._canvas.remove_item_graphic(item_id)

        self._item_list.refresh()
        self._property_panel.set_item(None)
        self._mark_modified()
        self._status_bar.showMessage(f"Deleted {len(selected_ids)} item(s)")

    def _select_all(self) -> None:
        """Select all items."""
        for item in self._document.items:
            self._canvas.select_item(item.id, clear_others=False)

    # =========================================================================
    # Viewport Capture
    # =========================================================================

    def _capture_current_viewport(self) -> None:
        """Capture the current 3D viewport and update viewport items."""
        if self._main_window is None:
            QMessageBox.information(
                self, "No Viewer",
                "No main viewer connected. Open Layout Composer from the main window."
            )
            return

        viewer = getattr(self._main_window, 'viewer_widget', None)
        if viewer is None:
            QMessageBox.warning(self, "Error", "Viewer widget not available")
            return

        try:
            # Get camera state
            renderer = getattr(viewer, 'renderer', None)
            camera_state = {}
            legend_state = {}

            if renderer:
                if hasattr(renderer, 'get_camera_info'):
                    camera_state = renderer.get_camera_info() or {}
                if hasattr(renderer, 'legend_manager'):
                    legend_mgr = renderer.legend_manager
                    if hasattr(legend_mgr, 'get_state'):
                        legend_state = legend_mgr.get_state() or {}

            # Capture screenshot
            plotter = getattr(renderer, 'plotter', None) if renderer else None
            pixmap = None

            if plotter:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    temp_path = f.name

                plotter.screenshot(temp_path)
                pixmap = QPixmap(temp_path)
                Path(temp_path).unlink(missing_ok=True)

            # Update all viewport items
            updated = 0
            for item in self._document.items:
                if isinstance(item, ViewportItem):
                    item.camera_state = camera_state
                    item.legend_state = legend_state

                    # Update graphics
                    if item.id in self._canvas._item_graphics:
                        graphic = self._canvas._item_graphics[item.id]
                        if pixmap and hasattr(graphic, 'set_viewport_pixmap'):
                            graphic.set_viewport_pixmap(pixmap)
                    updated += 1

            # Also update legend items
            for item in self._document.items:
                if isinstance(item, LegendItem):
                    item.legend_state = legend_state
                    self._canvas.update_item_graphic(item)

            self._mark_modified()
            self._status_bar.showMessage(f"Captured viewport, updated {updated} item(s)")

        except Exception as e:
            logger.error(f"Failed to capture viewport: {e}")
            QMessageBox.warning(self, "Error", f"Failed to capture viewport: {e}")

    # =========================================================================
    # Export
    # =========================================================================

    def _export(self, format_type: str) -> None:
        """Export the layout to the specified format."""
        dialog = ExportDialog(format_type, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        options = dialog.get_options()
        dpi = options.get("dpi", 300)

        # Get file path
        extensions = {"pdf": ".pdf", "png": ".png", "tiff": ".tiff"}
        ext = extensions.get(format_type, ".pdf")

        default_name = f"{self._document.name.replace(' ', '_')}_{dpi}dpi{ext}"
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export to {format_type.upper()}",
            str(Path.home() / default_name),
            f"{format_type.upper()} Files (*{ext})"
        )
        if not filepath:
            return

        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(ext)

        # Prepare metadata values
        metadata_values = {
            "project_name": self._document.name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": self._get_username(),
            "crs": self._get_crs_string(),
            "software_version": self._get_software_version(),
            "export_dpi": str(dpi),
            "page_size": f"{self._document.page.size.value} {self._document.page.orientation.value}",
        }

        # Get viewer for viewport capture
        viewer = None
        if self._main_window:
            viewer = getattr(self._main_window, 'viewer_widget', None)

        try:
            if format_type == "pdf":
                audit = export_pdf(
                    self._document, filepath, dpi, viewer, metadata_values
                )
            elif format_type == "png":
                audit = export_png(
                    self._document, filepath, dpi, viewer, metadata_values,
                    transparent=options.get("transparent", False)
                )
            elif format_type == "tiff":
                audit = export_tiff(
                    self._document, filepath, dpi, viewer, metadata_values,
                    compression=options.get("compression", "lzw")
                )
            else:
                raise ValueError(f"Unknown format: {format_type}")

            self._status_bar.showMessage(f"Exported to {filepath}")
            self.layout_exported.emit(str(filepath), format_type)

            QMessageBox.information(
                self, "Export Complete",
                f"Layout exported to:\n{filepath}\n\n"
                f"Audit record saved to:\n{filepath}.audit.json"
            )

        except Exception as e:
            logger.error(f"Export failed: {e}")
            QMessageBox.warning(self, "Export Failed", f"Failed to export: {e}")

    def _get_username(self) -> str:
        """Get current username."""
        import getpass
        return getpass.getuser()

    def _get_software_version(self) -> str:
        """Get GeoX software version."""
        try:
            from ...layout.layout_export import get_software_version
            return get_software_version()
        except Exception:
            return "GeoX"

    def _get_crs_string(self) -> str:
        """Get CRS string from current project."""
        try:
            if self._main_window:
                # Try to get CRS from app controller
                controller = getattr(self._main_window, 'app_controller', None)
                if controller:
                    crs = getattr(controller, 'crs', None)
                    if crs:
                        return str(crs)
                # Try to get from viewer/renderer
                viewer = getattr(self._main_window, 'viewer_widget', None)
                if viewer:
                    renderer = getattr(viewer, 'renderer', None)
                    if renderer:
                        crs = getattr(renderer, 'crs', None)
                        if crs:
                            return str(crs)
        except Exception as e:
            logger.debug(f"Could not get CRS: {e}")
        return "Local"

    # =========================================================================
    # View and Layout Operations
    # =========================================================================

    def _toggle_grid(self, checked: bool) -> None:
        """Toggle grid visibility."""
        self._canvas.set_grid_visible(checked)

    def _toggle_snap(self, checked: bool) -> None:
        """Toggle snap to grid."""
        self._canvas.set_snap_enabled(checked)

    def _show_page_setup(self) -> None:
        """Show page setup dialog."""
        dialog = PageSetupDialog(self._document.page, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._document.page = dialog.get_page_spec()
            self._refresh_canvas()
            self._mark_modified()

    def _apply_default_layout(self) -> None:
        """Apply the default layout template."""
        result = QMessageBox.question(
            self, "Apply Default Layout",
            "This will replace all items with the default layout. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if result == QMessageBox.StandardButton.Yes:
            self._document.create_default_layout()
            self._refresh_canvas()
            self._mark_modified()

    # =========================================================================
    # Signal Handlers
    # =========================================================================

    def _on_selection_changed(self, item_ids: List[str]) -> None:
        """Handle canvas selection change."""
        if len(item_ids) == 1:
            item = self._document.get_item(item_ids[0])
            self._property_panel.set_item(item)
            self._item_list.select_item(item_ids[0])
        else:
            self._property_panel.set_item(None)

    def _on_item_double_clicked(self, item_id: str) -> None:
        """Handle double-click on item."""
        item = self._document.get_item(item_id)
        if item:
            self._property_panel.set_item(item)
            self._property_dock.show()
            self._property_dock.raise_()

    def _on_list_item_selected(self, item_id: str) -> None:
        """Handle item selection from list."""
        self._canvas.select_item(item_id)
        item = self._document.get_item(item_id)
        self._property_panel.set_item(item)

    def _on_visibility_changed(self, item_id: str, visible: bool) -> None:
        """Handle visibility toggle from list."""
        item = self._document.get_item(item_id)
        if item:
            item.visible = visible
            self._canvas.update_item_graphic(item)
            self._mark_modified()

    def _on_item_deleted(self, item_id: str) -> None:
        """Handle item deletion from list."""
        self._document.remove_item(item_id)
        self._canvas.remove_item_graphic(item_id)
        self._property_panel.set_item(None)
        self._mark_modified()

    def _on_property_changed(self, item_id: str, property_name: str, value) -> None:
        """Handle property change from panel."""
        item = self._document.get_item(item_id)
        if item and hasattr(item, property_name):
            setattr(item, property_name, value)
            self._canvas.update_item_graphic(item)
            self._item_list.refresh()
            self._mark_modified()

    def _show_canvas_context_menu(self, scene_pos: QPointF) -> None:
        """Show context menu for canvas background."""
        menu = QMenu(self)
        menu.addAction("Add Viewport", lambda: self._add_item("viewport"))
        menu.addAction("Add Legend", lambda: self._add_item("legend"))
        menu.addAction("Add Text", lambda: self._add_item("text"))
        menu.addSeparator()
        menu.addAction("Fit Page", self._canvas.fit_page_in_view)
        menu.exec(self.cursor().pos())


class PageSetupDialog(QDialog):
    """Dialog for configuring page settings."""

    def __init__(self, page_spec: PageSpec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Page Setup")
        self.setMinimumWidth(300)

        self._page_spec = page_spec
        layout = QVBoxLayout(self)

        # Page size
        size_group = QGroupBox("Page Size")
        size_layout = QFormLayout(size_group)

        self._size_combo = QComboBox()
        for size in PageSize:
            self._size_combo.addItem(size.value, size)
        self._size_combo.setCurrentText(page_spec.size.value)
        size_layout.addRow("Size:", self._size_combo)

        self._orientation_combo = QComboBox()
        self._orientation_combo.addItem("Landscape", PageOrientation.LANDSCAPE)
        self._orientation_combo.addItem("Portrait", PageOrientation.PORTRAIT)
        idx = 0 if page_spec.orientation == PageOrientation.LANDSCAPE else 1
        self._orientation_combo.setCurrentIndex(idx)
        size_layout.addRow("Orientation:", self._orientation_combo)

        layout.addWidget(size_group)

        # Margins
        margin_group = QGroupBox("Margins (mm)")
        margin_layout = QFormLayout(margin_group)

        self._margin_left = QSpinBox()
        self._margin_left.setRange(0, 100)
        self._margin_left.setValue(int(page_spec.margin_left_mm))
        margin_layout.addRow("Left:", self._margin_left)

        self._margin_right = QSpinBox()
        self._margin_right.setRange(0, 100)
        self._margin_right.setValue(int(page_spec.margin_right_mm))
        margin_layout.addRow("Right:", self._margin_right)

        self._margin_top = QSpinBox()
        self._margin_top.setRange(0, 100)
        self._margin_top.setValue(int(page_spec.margin_top_mm))
        margin_layout.addRow("Top:", self._margin_top)

        self._margin_bottom = QSpinBox()
        self._margin_bottom.setRange(0, 100)
        self._margin_bottom.setValue(int(page_spec.margin_bottom_mm))
        margin_layout.addRow("Bottom:", self._margin_bottom)

        layout.addWidget(margin_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_page_spec(self) -> PageSpec:
        """Get the configured page specification."""
        return PageSpec(
            size=self._size_combo.currentData(),
            orientation=self._orientation_combo.currentData(),
            margin_left_mm=float(self._margin_left.value()),
            margin_right_mm=float(self._margin_right.value()),
            margin_top_mm=float(self._margin_top.value()),
            margin_bottom_mm=float(self._margin_bottom.value()),
        )


class ExportDialog(QDialog):
    """Dialog for export options."""

    def __init__(self, format_type: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Export to {format_type.upper()}")
        self.setMinimumWidth(300)

        self._format = format_type
        layout = QVBoxLayout(self)

        # DPI selection
        dpi_group = QGroupBox("Resolution")
        dpi_layout = QFormLayout(dpi_group)

        self._dpi_combo = QComboBox()
        for name, dpi in DPI_PRESETS.items():
            self._dpi_combo.addItem(f"{name.capitalize()} ({dpi} DPI)", dpi)
        self._dpi_combo.setCurrentIndex(2)  # Default to publication (300 DPI)
        dpi_layout.addRow("Preset:", self._dpi_combo)

        self._custom_dpi = QSpinBox()
        self._custom_dpi.setRange(72, 1200)
        self._custom_dpi.setValue(300)
        self._custom_dpi.setEnabled(False)
        dpi_layout.addRow("Custom DPI:", self._custom_dpi)

        self._use_custom = QCheckBox("Use custom DPI")
        self._use_custom.toggled.connect(self._custom_dpi.setEnabled)
        dpi_layout.addRow(self._use_custom)

        layout.addWidget(dpi_group)

        # Format-specific options
        if format_type == "png":
            options_group = QGroupBox("Options")
            options_layout = QFormLayout(options_group)

            self._transparent = QCheckBox("Transparent background")
            options_layout.addRow(self._transparent)

            layout.addWidget(options_group)

        elif format_type == "tiff":
            options_group = QGroupBox("Options")
            options_layout = QFormLayout(options_group)

            self._compression = QComboBox()
            self._compression.addItems(["LZW", "ZIP", "None"])
            options_layout.addRow("Compression:", self._compression)

            layout.addWidget(options_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_options(self) -> Dict:
        """Get the configured export options."""
        if self._use_custom.isChecked():
            dpi = self._custom_dpi.value()
        else:
            dpi = self._dpi_combo.currentData()

        options = {"dpi": dpi}

        if self._format == "png":
            options["transparent"] = self._transparent.isChecked()
        elif self._format == "tiff":
            options["compression"] = self._compression.currentText().lower()

        return options
