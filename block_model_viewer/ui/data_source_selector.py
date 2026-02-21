"""
Data Source Selector & Lineage Display Widgets
================================================

These widgets solve the "black box" problem by giving users:

1. **Explicit data source selection** - Users can SEE and CHOOSE which data
   they want to use (raw, composite, declustered, etc.)

2. **Visual lineage display** - A clear breadcrumb trail showing how data
   has been transformed

3. **Provenance warnings** - Alerts when data might not be appropriate
   for the current operation

Usage:
    # In any analysis panel that needs data
    from block_model_viewer.ui.data_source_selector import DataSourceSelector
    
    self.source_selector = DataSourceSelector(
        allowed_types=[DataSourceType.COMPOSITED, DataSourceType.DECLUSTERED],
        parent=self
    )
    self.source_selector.sourceChanged.connect(self._on_source_changed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QFrame, QToolButton, QMenu, QGroupBox, QSizePolicy,
    QScrollArea, QPushButton, QToolTip,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QIcon, QColor, QPainter, QPainterPath, QFont

from ..core.data_provenance import (
    DataProvenance, DataSourceType, TransformationStep,
    format_lineage_for_display, get_available_data_sources,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STYLING
# =============================================================================

SELECTOR_STYLESHEET = """
QComboBox {
    background-color: #1e1e1e;
    border: 2px solid #3a3a4a;
    border-radius: 6px;
    padding: 8px 12px;
    padding-right: 30px;
    color: #e0e0e0;
    font-size: 11pt;
    font-weight: 500;
    min-height: 24px;
}

QComboBox:hover {
    border-color: #4a9eff;
}

QComboBox:focus {
    border-color: #4a9eff;
    background-color: #252530;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 1px solid #3a3a4a;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    border: 1px solid #4a9eff;
    border-radius: 4px;
    selection-background-color: #2a4a6a;
    selection-color: white;
    padding: 4px;
}

QComboBox QAbstractItemView::item {
    padding: 8px 12px;
    min-height: 28px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #2a3a4a;
}
"""

LINEAGE_BANNER_STYLESHEET = """
QFrame#LineageBanner {
    background-color: #1a1a24;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 8px;
}

QLabel#LineageStep {
    color: #a0a0a0;
    font-size: 10pt;
}

QLabel#LineageStepCurrent {
    color: #4CAF50;
    font-weight: bold;
    font-size: 10pt;
}

QLabel#LineageArrow {
    color: #606070;
    font-size: 10pt;
}
"""

WARNING_BANNER_STYLESHEET = """
QFrame#WarningBanner {
    background-color: #3a2a1a;
    border: 1px solid #d4a546;
    border-radius: 6px;
    padding: 8px 12px;
}

QLabel#WarningText {
    color: #f5c842;
    font-size: 10pt;
}
"""


# =============================================================================
# DATA SOURCE SELECTOR WIDGET
# =============================================================================

class DataSourceSelector(QWidget):
    """
    A combobox-based selector that shows available data sources with
    their provenance information.
    
    Features:
    - Shows data source type with visual indicators
    - Displays row counts and transformation history
    - Allows filtering by allowed data types
    - Emits signals when selection changes
    
    Signals:
        sourceChanged(str, dict): Emitted when user selects a different source.
            Args: (source_key, source_info_dict)
    """
    
    sourceChanged = pyqtSignal(str, dict)  # (source_key, source_info)
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        allowed_types: Optional[List[DataSourceType]] = None,
        label: str = "Data Source:",
        show_row_count: bool = True,
    ):
        """
        Initialize the data source selector.
        
        Args:
            parent: Parent widget
            allowed_types: List of allowed DataSourceTypes. If None, all types allowed.
            label: Label text to show before the combobox
            show_row_count: Whether to show row counts in the dropdown
        """
        super().__init__(parent)
        
        self.allowed_types = allowed_types
        self.show_row_count = show_row_count
        self._sources: List[Dict[str, Any]] = []
        self._current_key: Optional[str] = None
        self._registry = None
        
        self._setup_ui(label)
        self.setStyleSheet(SELECTOR_STYLESHEET)
    
    def _setup_ui(self, label: str):
        """Build the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Label
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #b0b0b0; font-weight: 500;")
        layout.addWidget(self.label)
        
        # Combobox
        self.combo = QComboBox()
        self.combo.setMinimumWidth(280)
        self.combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)
        
        # Info button
        self.info_btn = QToolButton()
        self.info_btn.setText("ℹ")
        self.info_btn.setToolTip("View data lineage")
        self.info_btn.setStyleSheet("""
            QToolButton {
                background-color: #2a2a3a;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #80a0c0;
            }
            QToolButton:hover {
                background-color: #3a3a4a;
            }
        """)
        self.info_btn.clicked.connect(self._show_lineage_details)
        layout.addWidget(self.info_btn)
    
    def set_registry(self, registry) -> None:
        """
        Set the data registry to use for populating sources.
        
        Args:
            registry: DataRegistry instance
        """
        self._registry = registry
        self.refresh_sources()
    
    def refresh_sources(self) -> None:
        """Refresh the list of available data sources from the registry."""
        if not self._registry:
            return
        
        # Get all available sources
        all_sources = get_available_data_sources(self._registry)
        
        # Filter by allowed types
        if self.allowed_types:
            self._sources = [
                s for s in all_sources
                if s.get("type") in self.allowed_types or s.get("type") == DataSourceType.UNKNOWN
            ]
        else:
            self._sources = all_sources
        
        # Update combobox
        self.combo.blockSignals(True)
        self.combo.clear()
        
        for source in self._sources:
            display_text = self._format_source_display(source)
            self.combo.addItem(display_text, source.get("key"))
        
        # Restore selection if possible
        if self._current_key:
            for i, source in enumerate(self._sources):
                if source.get("key") == self._current_key:
                    self.combo.setCurrentIndex(i)
                    break
        
        self.combo.blockSignals(False)
        
        # Update info button state
        self.info_btn.setEnabled(len(self._sources) > 0)
    
    def _format_source_display(self, source: Dict[str, Any]) -> str:
        """Format a source for display in the combobox."""
        name = source.get("name", "Unknown")
        
        # Add type indicator
        data_type = source.get("type", DataSourceType.UNKNOWN)
        if isinstance(data_type, DataSourceType):
            type_indicator = "●"
            color = data_type.get_color()
        else:
            type_indicator = "○"
            color = "#808080"
        
        # Add row count if available
        row_count = source.get("row_count", 0)
        if self.show_row_count and row_count > 0:
            name += f"  ({row_count:,} rows)"
        
        return f"{type_indicator}  {name}"
    
    def _on_selection_changed(self, index: int) -> None:
        """Handle selection change."""
        if index < 0 or index >= len(self._sources):
            return
        
        source = self._sources[index]
        self._current_key = source.get("key")
        
        logger.info(f"Data source changed to: {source.get('name')}")
        self.sourceChanged.emit(self._current_key, source)
    
    def _show_lineage_details(self) -> None:
        """Show detailed lineage information for current selection."""
        if not self._sources:
            return
        
        current_idx = self.combo.currentIndex()
        if current_idx < 0 or current_idx >= len(self._sources):
            return
        
        source = self._sources[current_idx]
        provenance = source.get("provenance")
        
        if provenance and isinstance(provenance, DataProvenance):
            lineage_text = provenance.get_lineage_summary()
        else:
            lineage_text = f"No detailed provenance for {source.get('name')}"
        
        # Show as tooltip near the button
        QToolTip.showText(
            self.info_btn.mapToGlobal(self.info_btn.rect().bottomLeft()),
            f"<b>Data Lineage:</b><br>{lineage_text}",
            self.info_btn,
        )
    
    def get_current_source(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected source info."""
        if not self._sources:
            return None
        
        current_idx = self.combo.currentIndex()
        if current_idx < 0 or current_idx >= len(self._sources):
            return None
        
        return self._sources[current_idx]
    
    def get_current_key(self) -> Optional[str]:
        """Get the key of the currently selected source."""
        source = self.get_current_source()
        return source.get("key") if source else None
    
    def select_source(self, key: str) -> bool:
        """
        Programmatically select a source by key.

        Args:
            key: Source key to select

        Returns:
            True if source was found and selected
        """
        for i, source in enumerate(self._sources):
            if source.get("key") == key:
                self.combo.setCurrentIndex(i)
                return True
        return False

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(SELECTOR_STYLESHEET)


# =============================================================================
# DATA LINEAGE BANNER WIDGET
# =============================================================================

class DataLineageBanner(QFrame):
    """
    A visual banner showing the data lineage as a breadcrumb trail.
    
    Displays something like:
        [Raw Data] → [Composited] → [Declustered] → [Current]
    
    Each step is clickable to show details.
    """
    
    stepClicked = pyqtSignal(int, dict)  # (step_index, step_info)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LineageBanner")
        self._provenance: Optional[DataProvenance] = None
        self._steps: List[Dict[str, Any]] = []
        
        self._setup_ui()
        self.setStyleSheet(LINEAGE_BANNER_STYLESHEET)
    
    def _setup_ui(self):
        """Build the UI."""
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(12, 8, 12, 8)
        self.layout.setSpacing(6)
        
        # Title/icon
        self.title_label = QLabel("📊 Data Lineage:")
        self.title_label.setStyleSheet("color: #808090; font-weight: 500;")
        self.layout.addWidget(self.title_label)
        
        # Steps container
        self.steps_layout = QHBoxLayout()
        self.steps_layout.setSpacing(4)
        self.layout.addLayout(self.steps_layout)
        
        # Spacer
        self.layout.addStretch()
    
    def set_provenance(self, provenance: Optional[DataProvenance]) -> None:
        """
        Set the provenance to display.
        
        Args:
            provenance: DataProvenance object or None to clear
        """
        self._provenance = provenance
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the lineage display."""
        # Clear existing steps
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if not self._provenance:
            no_data = QLabel("No data loaded")
            no_data.setStyleSheet("color: #606070; font-style: italic;")
            self.steps_layout.addWidget(no_data)
            return
        
        # Format steps for display
        self._steps = format_lineage_for_display(self._provenance)
        
        for i, step in enumerate(self._steps):
            # Add arrow between steps (except before first)
            if i > 0:
                arrow = QLabel("→")
                arrow.setObjectName("LineageArrow")
                self.steps_layout.addWidget(arrow)
            
            # Create step label
            step_label = self._create_step_label(step, i)
            self.steps_layout.addWidget(step_label)
    
    def _create_step_label(self, step: Dict[str, Any], index: int) -> QLabel:
        """Create a label for a lineage step."""
        name = step.get("name", "Unknown")
        is_current = step.get("is_current", False)
        color = step.get("color", "#808080")
        
        label = QPushButton(name)
        label.setObjectName("LineageStepCurrent" if is_current else "LineageStep")
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        label.setFlat(True)
        
        # Style based on whether current
        if is_current:
            label.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color}20;
                    border: 1px solid {color};
                    border-radius: 4px;
                    padding: 4px 8px;
                    color: {color};
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}40;
                }}
            """)
        else:
            label.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid #3a3a4a;
                    border-radius: 4px;
                    padding: 4px 8px;
                    color: #909090;
                }}
                QPushButton:hover {{
                    background-color: #2a2a3a;
                    color: {color};
                }}
            """)
        
        # Connect click
        label.clicked.connect(lambda checked, idx=index: self._on_step_clicked(idx))
        
        # Tooltip with details
        tooltip_parts = [f"<b>{name}</b>"]
        if step.get("details"):
            tooltip_parts.append(f"<br>{step['details']}")
        if step.get("timestamp"):
            tooltip_parts.append(f"<br><i>{step['timestamp']}</i>")
        if step.get("row_count_after"):
            tooltip_parts.append(f"<br>Rows: {step['row_count_after']:,}")
        
        label.setToolTip("".join(tooltip_parts))
        
        return label
    
    def _on_step_clicked(self, index: int) -> None:
        """Handle step click."""
        if index < len(self._steps):
            self.stepClicked.emit(index, self._steps[index])


# =============================================================================
# WARNING BANNER WIDGET
# =============================================================================

class DataWarningBanner(QFrame):
    """
    A warning banner that appears when data provenance might be inappropriate.
    
    For example:
    - Using raw data for kriging (should use composites)
    - Using non-declustered data for variograms
    - Missing transformations in the chain
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("WarningBanner")
        self._setup_ui()
        self.setStyleSheet(WARNING_BANNER_STYLESHEET)
        self.hide()  # Hidden by default
    
    def _setup_ui(self):
        """Build the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Warning icon
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("font-size: 14pt;")
        layout.addWidget(icon_label)
        
        # Warning text
        self.text_label = QLabel()
        self.text_label.setObjectName("WarningText")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)
        
        # Dismiss button
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedSize(24, 24)
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #a08030;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #f5c842;
            }
        """)
        dismiss_btn.clicked.connect(self.hide)
        layout.addWidget(dismiss_btn)
    
    def show_warning(self, message: str) -> None:
        """Show a warning message."""
        self.text_label.setText(message)
        self.show()
    
    def check_provenance_for_panel(
        self,
        provenance: Optional[DataProvenance],
        panel_type: str,
    ) -> None:
        """
        Check if the provenance is appropriate for a specific panel type.
        
        Args:
            provenance: Current data provenance
            panel_type: Type of panel (e.g., "variogram", "kriging", "classification")
        """
        if not provenance:
            self.show_warning("No data provenance available. Unable to verify data source.")
            return
        
        warnings = []
        current_type = provenance.get_current_type()
        
        if panel_type == "variogram":
            # Variograms should use composited data
            if current_type == DataSourceType.RAW_ASSAYS:
                warnings.append(
                    "Using raw assay data. Consider compositing first for more reliable variograms."
                )
            # Preferably declustered
            if not provenance.has_transformation("declustering"):
                warnings.append(
                    "Data has not been declustered. Clustered data may bias variogram estimation."
                )
        
        elif panel_type == "kriging":
            # Kriging should use composited data
            if current_type == DataSourceType.RAW_ASSAYS:
                warnings.append(
                    "Using raw assay data for kriging. This may cause support effect issues."
                )
        
        elif panel_type == "classification":
            # Classification should use kriging/estimation results
            if current_type not in [
                DataSourceType.KRIGING_ESTIMATE,
                DataSourceType.SIMPLE_KRIGING_ESTIMATE,
                DataSourceType.INDICATOR_KRIGING_ESTIMATE,
                DataSourceType.SGSIM_SIMULATION,
            ]:
                warnings.append(
                    "Classification typically uses estimation results (kriging variance) for distance metrics."
                )
        
        elif panel_type == "resource_summary":
            # Resource summaries should use classified block models
            if current_type != DataSourceType.CLASSIFIED_BLOCK_MODEL:
                warnings.append(
                    "Resource summary is typically computed from classified block models."
                )
        
        if warnings:
            self.show_warning("\n".join(warnings))
        else:
            self.hide()


# =============================================================================
# COMBINED DATA SOURCE PANEL
# =============================================================================

class DataSourcePanel(QGroupBox):
    """
    A complete panel for data source selection with lineage display.
    
    Combines:
    - DataSourceSelector (dropdown)
    - DataLineageBanner (breadcrumb trail)
    - DataWarningBanner (warnings)
    
    This is the main widget that analysis panels should embed.
    """
    
    sourceChanged = pyqtSignal(str, dict)
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "Data Source",
        allowed_types: Optional[List[DataSourceType]] = None,
        panel_type: str = "generic",
    ):
        """
        Initialize the data source panel.
        
        Args:
            parent: Parent widget
            title: Group box title
            allowed_types: Allowed data source types
            panel_type: Type of panel for provenance checking
        """
        super().__init__(title, parent)
        
        self.panel_type = panel_type
        self._registry = None
        
        self._setup_ui(allowed_types)
        self._apply_styling()
    
    def _setup_ui(self, allowed_types):
        """Build the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Warning banner (hidden by default)
        self.warning_banner = DataWarningBanner()
        layout.addWidget(self.warning_banner)
        
        # Source selector
        self.selector = DataSourceSelector(
            allowed_types=allowed_types,
            label="Select Data:",
        )
        self.selector.sourceChanged.connect(self._on_source_changed)
        layout.addWidget(self.selector)
        
        # Lineage banner
        self.lineage_banner = DataLineageBanner()
        layout.addWidget(self.lineage_banner)
    
    def _apply_styling(self):
        """Apply styling to the panel."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #4fc3f7;
                border: 1px solid #3a3a4a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
    
    def set_registry(self, registry) -> None:
        """Set the data registry."""
        self._registry = registry
        self.selector.set_registry(registry)
    
    def refresh(self) -> None:
        """Refresh the panel."""
        self.selector.refresh_sources()
    
    def _on_source_changed(self, key: str, source_info: dict) -> None:
        """Handle source change."""
        # Update lineage banner
        provenance = source_info.get("provenance")
        self.lineage_banner.set_provenance(provenance)
        
        # Check for warnings
        self.warning_banner.check_provenance_for_panel(provenance, self.panel_type)
        
        # Emit signal
        self.sourceChanged.emit(key, source_info)
    
    def get_current_source(self) -> Optional[Dict[str, Any]]:
        """Get current selected source."""
        return self.selector.get_current_source()
    
    def get_current_key(self) -> Optional[str]:
        """Get current selected key."""
        return self.selector.get_current_key()


# =============================================================================
# QUICK STATUS INDICATOR
# =============================================================================

class DataSourceIndicator(QLabel):
    """
    A small indicator label showing current data source.
    
    Useful for displaying in panel headers or status bars.
    Shows something like: "📊 Composited (2,450 rows)"
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_styling()
        self.setText("No data")
    
    def _setup_styling(self):
        """Apply styling."""
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a24;
                border: 1px solid #2a2a3a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #a0a0a0;
                font-size: 9pt;
            }
        """)
    
    def update_from_provenance(self, provenance: Optional[DataProvenance]) -> None:
        """Update display from provenance."""
        if not provenance:
            self.setText("📊 No data")
            return
        
        current_type = provenance.get_current_type()
        name = current_type.get_display_name()
        color = current_type.get_color()
        
        row_text = ""
        if provenance.row_count:
            row_text = f" ({provenance.row_count:,} rows)"
        
        self.setText(f"📊 {name}{row_text}")
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color}15;
                border: 1px solid {color}40;
                border-radius: 4px;
                padding: 4px 8px;
                color: {color};
                font-size: 9pt;
            }}
        """)
    
    def update_from_source(self, source_info: Dict[str, Any]) -> None:
        """Update display from source info dict."""
        if not source_info:
            self.setText("📊 No data")
            return
        
        name = source_info.get("name", "Unknown")
        data_type = source_info.get("type", DataSourceType.UNKNOWN)
        row_count = source_info.get("row_count", 0)
        
        if isinstance(data_type, DataSourceType):
            color = data_type.get_color()
        else:
            color = "#808080"
        
        row_text = f" ({row_count:,})" if row_count else ""
        
        self.setText(f"📊 {name}{row_text}")
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color}15;
                border: 1px solid {color}40;
                border-radius: 4px;
                padding: 4px 8px;
                color: {color};
                font-size: 9pt;
            }}
        """)

