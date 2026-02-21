"""
Layout Property Panel for GeoX Layout Composer.

Provides context-sensitive property editing for layout items.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from ..modern_styles import get_theme_colors, ModernColors
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QScrollArea,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QTextEdit, QColorDialog, QPushButton, QLabel, QHBoxLayout
)

from ...layout.layout_document import (
    LayoutItem, ViewportItem, LegendItem, ScaleBarItem,
    NorthArrowItem, TextItem, ImageItem, MetadataItem
)

logger = logging.getLogger(__name__)


class LayoutPropertyPanel(QWidget):
    """
    Property panel for editing layout item properties.

    Shows context-appropriate editors based on the selected item type.
    """

    # Signal emitted when a property changes
    property_changed = pyqtSignal(str, str, object)  # item_id, property_name, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._item: Optional[LayoutItem] = None
        self._updating = False

        self._setup_ui()



    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()

    def _setup_ui(self) -> None:
        """Setup the UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Scroll area for properties
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #2d2d30;
                border: none;
            }
            QGroupBox {
                background-color: #2d2d30;
                color: #f0f0f0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel {
                color: #f0f0f0;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border-color: #007acc;
            }
            QCheckBox {
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #505050;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self._content)
        main_layout.addWidget(scroll)

        # Placeholder label
        self._no_selection_label = QLabel("No item selected")
        self._no_selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_selection_label.setStyleSheet("color: #808080;")
        self._content_layout.addWidget(self._no_selection_label)

    def set_item(self, item: Optional[LayoutItem]) -> None:
        """Set the item to edit."""
        self._item = item
        self._rebuild_editors()

    def _rebuild_editors(self) -> None:
        """Rebuild the property editors for the current item."""
        # Clear existing widgets
        while self._content_layout.count() > 0:
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self._item is None:
            label = QLabel("No item selected")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #808080;")
            self._content_layout.addWidget(label)
            return

        # Add common properties group
        self._add_common_properties()

        # Add type-specific properties
        if isinstance(self._item, ViewportItem):
            self._add_viewport_properties()
        elif isinstance(self._item, LegendItem):
            self._add_legend_properties()
        elif isinstance(self._item, ScaleBarItem):
            self._add_scale_bar_properties()
        elif isinstance(self._item, NorthArrowItem):
            self._add_north_arrow_properties()
        elif isinstance(self._item, TextItem):
            self._add_text_properties()
        elif isinstance(self._item, ImageItem):
            self._add_image_properties()
        elif isinstance(self._item, MetadataItem):
            self._add_metadata_properties()

        # Stretch at end
        self._content_layout.addStretch()

    def _add_common_properties(self) -> None:
        """Add common property editors."""
        group = QGroupBox("General")
        layout = QFormLayout(group)

        # Name
        self._name_edit = QLineEdit(self._item.name)
        self._name_edit.textChanged.connect(
            lambda v: self._emit_change("name", v)
        )
        layout.addRow("Name:", self._name_edit)

        # Type (read-only)
        type_label = QLabel(self._item.item_type.replace("_", " ").title())
        layout.addRow("Type:", type_label)

        # Visible
        self._visible_check = QCheckBox()
        self._visible_check.setChecked(self._item.visible)
        self._visible_check.toggled.connect(
            lambda v: self._emit_change("visible", v)
        )
        layout.addRow("Visible:", self._visible_check)

        # Locked
        self._locked_check = QCheckBox()
        self._locked_check.setChecked(self._item.locked)
        self._locked_check.toggled.connect(
            lambda v: self._emit_change("locked", v)
        )
        layout.addRow("Locked:", self._locked_check)

        self._content_layout.addWidget(group)

        # Position & Size group
        pos_group = QGroupBox("Position & Size (mm)")
        pos_layout = QFormLayout(pos_group)

        self._x_spin = QDoubleSpinBox()
        self._x_spin.setRange(-1000, 1000)
        self._x_spin.setDecimals(1)
        self._x_spin.setValue(self._item.x_mm)
        self._x_spin.valueChanged.connect(
            lambda v: self._emit_change("x_mm", v)
        )
        pos_layout.addRow("X:", self._x_spin)

        self._y_spin = QDoubleSpinBox()
        self._y_spin.setRange(-1000, 1000)
        self._y_spin.setDecimals(1)
        self._y_spin.setValue(self._item.y_mm)
        self._y_spin.valueChanged.connect(
            lambda v: self._emit_change("y_mm", v)
        )
        pos_layout.addRow("Y:", self._y_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1, 1000)
        self._width_spin.setDecimals(1)
        self._width_spin.setValue(self._item.width_mm)
        self._width_spin.valueChanged.connect(
            lambda v: self._emit_change("width_mm", v)
        )
        pos_layout.addRow("Width:", self._width_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(1, 1000)
        self._height_spin.setDecimals(1)
        self._height_spin.setValue(self._item.height_mm)
        self._height_spin.valueChanged.connect(
            lambda v: self._emit_change("height_mm", v)
        )
        pos_layout.addRow("Height:", self._height_spin)

        self._content_layout.addWidget(pos_group)

    def _add_viewport_properties(self) -> None:
        """Add viewport-specific properties."""
        item: ViewportItem = self._item
        group = QGroupBox("Viewport")
        layout = QFormLayout(group)

        # Background color
        self._add_color_picker(
            layout, "Background:", item.background_color, "background_color"
        )

        # Show axes
        axes_check = QCheckBox()
        axes_check.setChecked(item.show_axes)
        axes_check.toggled.connect(lambda v: self._emit_change("show_axes", v))
        layout.addRow("Show Axes:", axes_check)

        # Camera info (read-only)
        if item.camera_state:
            cam_label = QLabel("Camera captured")
            cam_label.setStyleSheet("color: #50c050;")
        else:
            cam_label = QLabel("No camera state")
            cam_label.setStyleSheet("color: #c05050;")
        layout.addRow("Status:", cam_label)

        self._content_layout.addWidget(group)

    def _add_legend_properties(self) -> None:
        """Add legend-specific properties."""
        item: LegendItem = self._item
        group = QGroupBox("Legend")
        layout = QFormLayout(group)

        # Orientation
        orient_combo = QComboBox()
        orient_combo.addItems(["vertical", "horizontal"])
        orient_combo.setCurrentText(item.orientation)
        orient_combo.currentTextChanged.connect(
            lambda v: self._emit_change("orientation", v)
        )
        layout.addRow("Orientation:", orient_combo)

        # Font family
        font_combo = QComboBox()
        font_combo.addItems(["Segoe UI", "Arial", "Helvetica", "Times New Roman", "Courier New"])
        font_combo.setCurrentText(getattr(item, 'font_family', "Segoe UI"))
        font_combo.currentTextChanged.connect(
            lambda v: self._emit_change("font_family", v)
        )
        layout.addRow("Font:", font_combo)

        # Font size
        font_spin = QSpinBox()
        font_spin.setRange(6, 24)
        font_spin.setValue(item.font_size)
        font_spin.valueChanged.connect(
            lambda v: self._emit_change("font_size", v)
        )
        layout.addRow("Font Size:", font_spin)

        # Show title
        title_check = QCheckBox()
        title_check.setChecked(item.show_title)
        title_check.toggled.connect(lambda v: self._emit_change("show_title", v))
        layout.addRow("Show Title:", title_check)

        # Colors
        self._add_color_picker(layout, "Background:", item.background_color, "background_color")
        self._add_color_picker(layout, "Text Color:", item.text_color, "text_color")

        self._content_layout.addWidget(group)

    def _add_scale_bar_properties(self) -> None:
        """Add scale bar-specific properties."""
        item: ScaleBarItem = self._item
        group = QGroupBox("Scale Bar")
        layout = QFormLayout(group)

        # Units
        units_combo = QComboBox()
        units_combo.addItems(["m", "km", "ft", "mi"])
        units_combo.setCurrentText(item.units)
        units_combo.currentTextChanged.connect(
            lambda v: self._emit_change("units", v)
        )
        layout.addRow("Units:", units_combo)

        # Style
        style_combo = QComboBox()
        style_combo.addItems(["alternating", "single", "stepped"])
        style_combo.setCurrentText(item.style)
        style_combo.currentTextChanged.connect(
            lambda v: self._emit_change("style", v)
        )
        layout.addRow("Style:", style_combo)

        # Segments
        seg_spin = QSpinBox()
        seg_spin.setRange(1, 10)
        seg_spin.setValue(item.num_segments)
        seg_spin.valueChanged.connect(
            lambda v: self._emit_change("num_segments", v)
        )
        layout.addRow("Segments:", seg_spin)

        # Font size
        font_spin = QSpinBox()
        font_spin.setRange(6, 24)
        font_spin.setValue(item.font_size)
        font_spin.valueChanged.connect(
            lambda v: self._emit_change("font_size", v)
        )
        layout.addRow("Font Size:", font_spin)

        self._content_layout.addWidget(group)

    def _add_north_arrow_properties(self) -> None:
        """Add north arrow-specific properties."""
        item: NorthArrowItem = self._item
        group = QGroupBox("North Arrow")
        layout = QFormLayout(group)

        # Style
        style_combo = QComboBox()
        style_combo.addItems(["simple", "compass"])
        style_combo.setCurrentText(item.style)
        style_combo.currentTextChanged.connect(
            lambda v: self._emit_change("style", v)
        )
        layout.addRow("Style:", style_combo)

        # Show label
        label_check = QCheckBox()
        label_check.setChecked(item.show_label)
        label_check.toggled.connect(lambda v: self._emit_change("show_label", v))
        layout.addRow("Show Label:", label_check)

        # Label text
        label_edit = QLineEdit(item.label_text)
        label_edit.textChanged.connect(
            lambda v: self._emit_change("label_text", v)
        )
        layout.addRow("Label:", label_edit)

        # Colors
        self._add_color_picker(layout, "Fill Color:", item.fill_color, "fill_color")
        self._add_color_picker(layout, "Outline:", item.outline_color, "outline_color")

        self._content_layout.addWidget(group)

    def _add_text_properties(self) -> None:
        """Add text-specific properties."""
        item: TextItem = self._item
        group = QGroupBox("Text")
        layout = QFormLayout(group)

        # Text content
        text_edit = QTextEdit()
        text_edit.setPlainText(item.text)
        text_edit.setMaximumHeight(100)
        text_edit.textChanged.connect(
            lambda: self._emit_change("text", text_edit.toPlainText())
        )
        layout.addRow("Text:", text_edit)

        # Font family
        font_combo = QComboBox()
        font_combo.addItems(["Segoe UI", "Arial", "Helvetica", "Times New Roman", "Courier New"])
        font_combo.setCurrentText(item.font_family)
        font_combo.currentTextChanged.connect(
            lambda v: self._emit_change("font_family", v)
        )
        layout.addRow("Font:", font_combo)

        # Font size
        font_spin = QSpinBox()
        font_spin.setRange(6, 72)
        font_spin.setValue(item.font_size)
        font_spin.valueChanged.connect(
            lambda v: self._emit_change("font_size", v)
        )
        layout.addRow("Size:", font_spin)

        # Bold
        bold_check = QCheckBox()
        bold_check.setChecked(item.font_bold)
        bold_check.toggled.connect(lambda v: self._emit_change("font_bold", v))
        layout.addRow("Bold:", bold_check)

        # Italic
        italic_check = QCheckBox()
        italic_check.setChecked(item.font_italic)
        italic_check.toggled.connect(lambda v: self._emit_change("font_italic", v))
        layout.addRow("Italic:", italic_check)

        # Alignment
        align_combo = QComboBox()
        align_combo.addItems(["left", "center", "right"])
        align_combo.setCurrentText(item.alignment)
        align_combo.currentTextChanged.connect(
            lambda v: self._emit_change("alignment", v)
        )
        layout.addRow("Alignment:", align_combo)

        # Colors
        self._add_color_picker(layout, "Text Color:", item.text_color, "text_color")

        self._content_layout.addWidget(group)

    def _add_image_properties(self) -> None:
        """Add image-specific properties."""
        item: ImageItem = self._item
        group = QGroupBox("Image")
        layout = QFormLayout(group)

        # Path (read-only)
        path_label = QLabel(item.image_path or "(embedded)" if item.image_data_base64 else "(no image)")
        path_label.setWordWrap(True)
        layout.addRow("Path:", path_label)

        # Maintain aspect
        aspect_check = QCheckBox()
        aspect_check.setChecked(item.maintain_aspect)
        aspect_check.toggled.connect(
            lambda v: self._emit_change("maintain_aspect", v)
        )
        layout.addRow("Keep Aspect:", aspect_check)

        # Opacity
        opacity_spin = QDoubleSpinBox()
        opacity_spin.setRange(0.0, 1.0)
        opacity_spin.setSingleStep(0.1)
        opacity_spin.setDecimals(2)
        opacity_spin.setValue(item.opacity)
        opacity_spin.valueChanged.connect(
            lambda v: self._emit_change("opacity", v)
        )
        layout.addRow("Opacity:", opacity_spin)

        self._content_layout.addWidget(group)

    def _add_metadata_properties(self) -> None:
        """Add metadata-specific properties."""
        item: MetadataItem = self._item
        group = QGroupBox("Metadata")
        layout = QFormLayout(group)

        # Fields (simplified - just show count)
        fields_label = QLabel(f"{len(item.fields)} fields configured")
        layout.addRow("Fields:", fields_label)

        # Font size
        font_spin = QSpinBox()
        font_spin.setRange(6, 18)
        font_spin.setValue(item.font_size)
        font_spin.valueChanged.connect(
            lambda v: self._emit_change("font_size", v)
        )
        layout.addRow("Font Size:", font_spin)

        # Show labels
        labels_check = QCheckBox()
        labels_check.setChecked(item.show_labels)
        labels_check.toggled.connect(
            lambda v: self._emit_change("show_labels", v)
        )
        layout.addRow("Show Labels:", labels_check)

        # Text color
        self._add_color_picker(layout, "Text Color:", item.text_color, "text_color")

        self._content_layout.addWidget(group)

    def _add_color_picker(self, layout: QFormLayout, label: str,
                          current_color: str, property_name: str) -> None:
        """Add a color picker row."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        color_btn = QPushButton()
        color_btn.setFixedSize(24, 24)
        color_btn.setStyleSheet(
            f"background-color: {current_color}; border: 1px solid #505050;"
        )

        def pick_color():
            color = QColorDialog.getColor(
                QColor(current_color), self, f"Select {label}"
            )
            if color.isValid():
                color_btn.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #505050;"
                )
                self._emit_change(property_name, color.name())

        color_btn.clicked.connect(pick_color)
        row_layout.addWidget(color_btn)

        color_label = QLabel(current_color)
        color_label.setStyleSheet("color: #a0a0a0;")
        row_layout.addWidget(color_label)
        row_layout.addStretch()

        layout.addRow(label, row_widget)

    def _emit_change(self, property_name: str, value: Any) -> None:
        """Emit property change signal."""
        if self._updating or self._item is None:
            return
        self.property_changed.emit(self._item.id, property_name, value)
