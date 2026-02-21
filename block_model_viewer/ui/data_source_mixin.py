"""
Data Source Selection Mixin
============================

A mixin class that provides consistent data source selection and provenance
tracking across all analysis panels.

This replaces the duplicated data source selection code in:
- VariogramPanel
- KrigingPanel
- SimpleKrigingPanel
- CoKrigingPanel
- UniversalKrigingPanel
- IndicatorKrigingPanel
- SGSIMPanel
- DeclusteringPanel

Usage:
    class MyPanel(BaseAnalysisPanel, DataSourceMixin):
        def __init__(self, parent=None):
            super().__init__(parent)
            # Initialize the mixin
            self.init_data_source_selection(self.config_layout)
            
        def _on_source_selected(self, key: str, source_info: dict):
            # Handle data source change
            self.update_with_data(source_info.get('data'))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup, QFrame, QToolTip,
    QPushButton, QSizePolicy, QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

if TYPE_CHECKING:
    from ..core.data_registry import DataRegistry

logger = logging.getLogger(__name__)


class DataSourceMixin:
    """
    Mixin class providing data source selection and provenance tracking.
    
    Features:
    - Dropdown to select data source (raw, composites, declustered)
    - Shows row counts and data lineage
    - Emits signal when source changes
    - Validates data source for specific panel types
    
    To use this mixin:
    1. Inherit from both BaseAnalysisPanel and DataSourceMixin
    2. Call init_data_source_selection() after setting up the main layout
    3. Connect to data_source_changed signal to react to changes
    4. Override get_allowed_source_types() to restrict available sources
    """
    
    # Signal emitted when data source changes
    data_source_changed = pyqtSignal(str, dict)  # (key, source_info)
    
    def init_data_source_selection(
        self,
        parent_layout: QVBoxLayout,
        panel_type: str = "generic",
        show_lineage: bool = True,
    ) -> QGroupBox:
        """
        Initialize data source selection UI.
        
        Args:
            parent_layout: Layout to add the data source UI to
            panel_type: Type of panel (for validation warnings)
            show_lineage: Whether to show lineage breadcrumbs
            
        Returns:
            The created QGroupBox widget
        """
        self._ds_panel_type = panel_type
        self._ds_show_lineage = show_lineage
        self._ds_current_key: Optional[str] = None
        self._ds_sources: List[Dict[str, Any]] = []
        
        # Create group box
        self._ds_group = QGroupBox("📊 Data Source")
        self._ds_group.setStyleSheet("""
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
        
        layout = QVBoxLayout(self._ds_group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(8)
        
        # Warning banner (hidden by default)
        self._ds_warning_frame = QFrame()
        self._ds_warning_frame.setObjectName("WarningBanner")
        self._ds_warning_frame.setStyleSheet("""
            QFrame#WarningBanner {
                background-color: #3a2a1a;
                border: 1px solid #d4a546;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        warning_layout = QHBoxLayout(self._ds_warning_frame)
        warning_layout.setContentsMargins(8, 4, 8, 4)
        
        self._ds_warning_icon = QLabel("⚠️")
        self._ds_warning_text = QLabel("")
        self._ds_warning_text.setStyleSheet("color: #f5c842; font-size: 10pt;")
        self._ds_warning_text.setWordWrap(True)
        warning_layout.addWidget(self._ds_warning_icon)
        warning_layout.addWidget(self._ds_warning_text, 1)
        
        self._ds_warning_frame.hide()
        layout.addWidget(self._ds_warning_frame)
        
        # Source selector row
        selector_layout = QHBoxLayout()
        selector_layout.setSpacing(8)
        
        self._ds_combo = QComboBox()
        self._ds_combo.setMinimumWidth(200)
        self._ds_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._ds_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 2px solid #3a3a4a;
                border-radius: 6px;
                padding: 6px 10px;
                padding-right: 25px;
                color: #e0e0e0;
                font-size: 10pt;
            }
            QComboBox:hover {
                border-color: #4a9eff;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                border: 1px solid #4a9eff;
                selection-background-color: #2a4a6a;
            }
        """)
        self._ds_combo.currentIndexChanged.connect(self._on_ds_selection_changed)
        selector_layout.addWidget(self._ds_combo)
        
        # Refresh button
        self._ds_refresh_btn = QPushButton("🔄")
        self._ds_refresh_btn.setFixedSize(28, 28)
        self._ds_refresh_btn.setToolTip("Refresh available data sources")
        self._ds_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a3a;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a3a4a;
            }
        """)
        self._ds_refresh_btn.clicked.connect(self.refresh_data_sources)
        selector_layout.addWidget(self._ds_refresh_btn)
        
        layout.addLayout(selector_layout)
        
        # Status/lineage label
        self._ds_status_label = QLabel("")
        self._ds_status_label.setStyleSheet("color: #808090; font-size: 9pt;")
        self._ds_status_label.setWordWrap(True)
        layout.addWidget(self._ds_status_label)
        
        # Lineage breadcrumb (if enabled)
        if show_lineage:
            self._ds_lineage_frame = QFrame()
            self._ds_lineage_frame.setStyleSheet("""
                QFrame {
                    background-color: #1a1a24;
                    border: 1px solid #2a2a3a;
                    border-radius: 6px;
                    padding: 6px;
                }
            """)
            self._ds_lineage_layout = QHBoxLayout(self._ds_lineage_frame)
            self._ds_lineage_layout.setContentsMargins(8, 4, 8, 4)
            self._ds_lineage_layout.setSpacing(4)
            
            lineage_title = QLabel("Lineage:")
            lineage_title.setStyleSheet("color: #606070; font-size: 9pt;")
            self._ds_lineage_layout.addWidget(lineage_title)
            self._ds_lineage_layout.addStretch()
            
            layout.addWidget(self._ds_lineage_frame)
        else:
            self._ds_lineage_frame = None
        
        parent_layout.addWidget(self._ds_group)
        
        # Initial refresh (deferred to avoid issues during init)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.refresh_data_sources)
        
        return self._ds_group
    
    def get_allowed_source_types(self) -> Optional[List[str]]:
        """
        Override this method to restrict available source types.
        
        Returns:
            List of allowed source keys, or None for all sources.
            
        Example:
            return ["composites", "declustered"]  # Only transformed data
        """
        return None
    
    def refresh_data_sources(self) -> None:
        """Refresh the list of available data sources."""
        try:
            registry = self.get_registry() if hasattr(self, 'get_registry') else None
            if not registry:
                self._ds_sources = []
                self._update_ds_combo()
                return
            
            # Get drillhole data
            dh_data = registry.get_drillhole_data(copy_data=False)
            
            sources = []
            
            if dh_data:
                # Check for raw assays
                if "assays" in dh_data:
                    assays = dh_data["assays"]
                    if assays is not None and len(assays) > 0:
                        sources.append({
                            "key": "raw_assays",
                            "name": "Raw Assays",
                            "type": "raw",
                            "row_count": len(assays),
                            "icon": "○",
                            "color": "#808080",
                            "data_key": "assays",
                        })
                
                # Check for composites
                if "composites" in dh_data:
                    comp = dh_data["composites"]
                    if comp is not None and len(comp) > 0:
                        sources.append({
                            "key": "composites",
                            "name": "Composited Data",
                            "type": "composited",
                            "row_count": len(comp),
                            "icon": "●",
                            "color": "#2196F3",
                            "data_key": "composites",
                        })
            
            # Check for declustered data
            if registry.has_data("declustering_results"):
                declust = registry.get_data("declustering_results", copy_data=False)
                if declust and "weighted_dataframe" in declust:
                    df = declust["weighted_dataframe"]
                    sources.append({
                        "key": "declustered",
                        "name": "Declustered Data",
                        "type": "declustered",
                        "row_count": len(df) if df is not None else 0,
                        "icon": "◆",
                        "color": "#4CAF50",
                        "data_key": "declustering_results",
                    })
            
            # Filter by allowed types
            allowed = self.get_allowed_source_types()
            if allowed:
                sources = [s for s in sources if s["key"] in allowed]
            
            self._ds_sources = sources
            self._update_ds_combo()
            
            # Get provenance if available
            self._update_provenance_display()
            
        except Exception as e:
            logger.warning(f"Failed to refresh data sources: {e}")
            self._ds_sources = []
            self._update_ds_combo()
    
    def _update_ds_combo(self) -> None:
        """Update the combo box with available sources."""
        if not hasattr(self, '_ds_combo'):
            return
            
        self._ds_combo.blockSignals(True)
        self._ds_combo.clear()
        
        if not self._ds_sources:
            self._ds_combo.addItem("No data available")
            self._ds_combo.setEnabled(False)
            self._ds_status_label.setText("Load drillhole data to begin analysis")
        else:
            self._ds_combo.setEnabled(True)
            for source in self._ds_sources:
                icon = source.get("icon", "○")
                name = source.get("name", "Unknown")
                count = source.get("row_count", 0)
                display = f"{icon}  {name}  ({count:,} rows)"
                self._ds_combo.addItem(display, source.get("key"))
            
            # Auto-select best source (prefer composites > declustered > raw)
            preferred_order = ["declustered", "composites", "raw_assays"]
            for pref in preferred_order:
                for i, source in enumerate(self._ds_sources):
                    if source["key"] == pref:
                        self._ds_combo.setCurrentIndex(i)
                        break
                else:
                    continue
                break
        
        self._ds_combo.blockSignals(False)
        
        # Trigger initial selection callback
        if self._ds_sources:
            self._on_ds_selection_changed(self._ds_combo.currentIndex())
    
    def _on_ds_selection_changed(self, index: int) -> None:
        """Handle data source selection change."""
        if index < 0 or index >= len(self._ds_sources):
            return
        
        source = self._ds_sources[index]
        self._ds_current_key = source.get("key")
        
        # Update status
        color = source.get("color", "#808080")
        name = source.get("name", "Unknown")
        self._ds_status_label.setText(f"Using: {name}")
        self._ds_status_label.setStyleSheet(f"color: {color}; font-size: 9pt;")
        
        # Check for warnings
        self._check_source_warnings(source)
        
        # Update provenance display
        self._update_provenance_display()
        
        # Emit signal for panel to handle
        logger.info(f"Data source changed to: {name}")
        if hasattr(self, 'data_source_changed'):
            self.data_source_changed.emit(self._ds_current_key, source)
        
        # Call handler if defined
        if hasattr(self, '_on_data_source_changed'):
            self._on_data_source_changed(self._ds_current_key, source)
    
    def _check_source_warnings(self, source: Dict[str, Any]) -> None:
        """Check if the selected source is appropriate for this panel type."""
        if not hasattr(self, '_ds_warning_frame'):
            return
        
        warnings = []
        source_type = source.get("type", "unknown")
        panel_type = getattr(self, '_ds_panel_type', 'generic')
        
        if panel_type == "variogram":
            if source_type == "raw":
                warnings.append(
                    "Using raw assay data. Consider compositing first for more reliable variograms."
                )
            if source_type != "declustered":
                warnings.append(
                    "Consider using declustered data to avoid clustering bias in variogram estimation."
                )
        
        elif panel_type == "kriging":
            if source_type == "raw":
                warnings.append(
                    "Using raw assay data for kriging. Support effect may cause issues. "
                    "Consider using composited data."
                )
        
        elif panel_type == "simulation":
            if source_type == "raw":
                warnings.append(
                    "Simulations should typically use composited or declustered data."
                )
        
        if warnings:
            self._ds_warning_text.setText(" ".join(warnings))
            self._ds_warning_frame.show()
        else:
            self._ds_warning_frame.hide()
    
    def _update_provenance_display(self) -> None:
        """Update the lineage breadcrumb display."""
        if not hasattr(self, '_ds_lineage_frame') or not self._ds_lineage_frame:
            return
        
        # Clear existing lineage items
        while self._ds_lineage_layout.count() > 2:  # Keep title and stretch
            item = self._ds_lineage_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        
        # Get provenance from registry
        try:
            registry = self.get_registry() if hasattr(self, 'get_registry') else None
            if not registry or not self._ds_current_key:
                return
            
            # Build simple lineage display
            lineage_parts = []
            
            # Add source file if known
            metadata = registry.get_drillhole_metadata()
            if metadata and metadata.source_file:
                lineage_parts.append(("📁", metadata.source_file.split("/")[-1].split("\\")[-1], "#808080"))
            else:
                lineage_parts.append(("📁", "Imported Data", "#808080"))
            
            # Add transformations based on current source
            current_source = next(
                (s for s in self._ds_sources if s["key"] == self._ds_current_key),
                None
            )
            
            if current_source:
                source_type = current_source.get("type", "unknown")
                
                if source_type in ["composited", "declustered"]:
                    lineage_parts.append(("→", "", "#606060"))
                    lineage_parts.append(("📊", "Composited", "#2196F3"))
                
                if source_type == "declustered":
                    lineage_parts.append(("→", "", "#606060"))
                    lineage_parts.append(("⚖️", "Declustered", "#4CAF50"))
            
            # Add labels
            for icon, text, color in lineage_parts:
                if text:
                    label = QLabel(f"{icon} {text}")
                else:
                    label = QLabel(icon)
                label.setStyleSheet(f"color: {color}; font-size: 9pt;")
                self._ds_lineage_layout.insertWidget(
                    self._ds_lineage_layout.count() - 1,  # Before stretch
                    label
                )
                
        except Exception as e:
            logger.debug(f"Failed to update provenance display: {e}")
    
    def get_selected_data(self) -> Optional[Any]:
        """
        Get the currently selected data.
        
        Returns:
            The DataFrame or data for the selected source, or None
        """
        if not self._ds_current_key:
            return None
        
        try:
            registry = self.get_registry() if hasattr(self, 'get_registry') else None
            if not registry:
                return None
            
            if self._ds_current_key in ["raw_assays", "composites"]:
                dh_data = registry.get_drillhole_data()
                if not dh_data:
                    return None
                
                if self._ds_current_key == "raw_assays":
                    return dh_data.get("assays")
                else:
                    return dh_data.get("composites")
            
            elif self._ds_current_key == "declustered":
                declust = registry.get_data("declustering_results")
                if declust:
                    return declust.get("weighted_dataframe")
                return None
            
        except Exception as e:
            logger.warning(f"Failed to get selected data: {e}")
            return None
    
    def get_selected_source_key(self) -> Optional[str]:
        """Get the key of the currently selected source."""
        return self._ds_current_key
    
    def get_selected_source_info(self) -> Optional[Dict[str, Any]]:
        """Get full info dict for the currently selected source."""
        if not self._ds_current_key:
            return None
        return next(
            (s for s in self._ds_sources if s["key"] == self._ds_current_key),
            None
        )

